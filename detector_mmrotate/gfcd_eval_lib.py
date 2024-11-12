import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import torch
import torchvision


def compute_loc_performance(gt_array, pred_array, eval_type, distance_tolerance, iou_threshold):
    # distance_matrix below doesn't work when preds is empty, so handle that first
    if len(pred_array) == 0:
        return [], [], gt_array.tolist()

    # Different matrix comparison depending on evaluating using center or IoU
    if eval_type == 'center':
        # Building distance matrix using Euclidean distance pixel space
        # multiplied by the UTM resolution (10 m per pixel)
        dist_mat = distance_matrix(pred_array, gt_array, p=2)
        dist_mat[dist_mat > distance_tolerance] = 99999
    elif eval_type == 'iou':
        # Compute pair-wise IoU between the current class targets and all predictions
        dist_mat = torchvision.ops.box_iou(gt_array, pred_array)
        dist_mat[dist_mat < iou_threshold] = 99999
        dist_mat = torch.transpose(dist_mat, 1, 0).cpu().detach().numpy()

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    if eval_type == 'center':
        tp_inds = [
            {"pred_idx": rows[ii], "gt_idx": cols[ii]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] < distance_tolerance
        ]
    elif eval_type == 'iou':
        tp_inds = [
            {"pred_idx": rows[ii], "gt_idx": cols[ii]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] > iou_threshold
        ]

    tp = [
        {'pred': pred_array[a['pred_idx']].tolist(), 'gt': gt_array[a['gt_idx']].tolist()}
        for a in tp_inds
    ]
    tp_inds_pred = set([a['pred_idx'] for a in tp_inds])
    tp_inds_gt = set([a['gt_idx'] for a in tp_inds])
    fp = [pred_array[i].tolist() for i in range(len(pred_array)) if i not in tp_inds_pred]
    fn = [gt_array[i].tolist() for i in range(len(gt_array)) if i not in tp_inds_gt]

    return tp, fp, fn


def metric_score(gt, pred, eval_type, distance_tolerance=20, iou_threshold=0.5):
    tp, fp, fn = [], [], []

    for scene_id in gt.keys():
        cur_tp, cur_fp, cur_fn = compute_loc_performance(gt[scene_id], pred[scene_id], eval_type, distance_tolerance, iou_threshold)
        tp += [{'scene_id': scene_id, 'pred': a['pred'], 'gt': a['gt']} for a in cur_tp]
        fp += [{'scene_id': scene_id, 'point': a} for a in cur_fp]
        fn += [{'scene_id': scene_id, 'point': a} for a in cur_fn]

    return len(tp), len(fp), len(fn)


class DetectF1Evaluator:
    def __init__(self, task, spec, detail_func=None, params=None):
        '''
        params: list of thresholds, one for each class.
        '''
        self.eval_type = spec.get('EvalType', 'center')
        self.iou_threshold = spec.get('EvalIOUThreshold', 0.5)
        self.distance_tolerance = spec.get('EvalDistanceTolerance', 20)
        self.task = task
        self.detail_func = detail_func

        # Set thresholds: each class can have multiple threshold options.
        num_classes = len(self.task['categories'])
        if params:
            self.thresholds = [[threshold] for threshold in params]
        else:
            self.thresholds = [[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] for _ in range(num_classes)]

        self.true_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_negatives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]

    def evaluate(self, gt_raw, pred_raw):

        # Compute F1 score for each class individually, at each of that class' score thresholds
        for cls_idx, cls_thresholds in enumerate(self.thresholds):

            if cls_idx == 0:
                continue

            # Filter out ground truth objects with the current class label
            # If the evaluation type is center-based, replace boxes with centers
            gt = {}
            for image_idx, target in enumerate(gt_raw):
                # Get the relevant boxes (i.e., matching cls).
                # Have to handle no-box case separately since labels length is >= 1.
                if len(target['boxes']) == 0:
                    boxes = target['boxes'].numpy()
                else:
                    boxes = target['boxes'][target['labels'] == cls_idx, :].numpy()

                if self.eval_type == 'center':
                    gt[image_idx] = np.stack([
                        (boxes[:, 0] + boxes[:, 2])/2,
                        (boxes[:, 1] + boxes[:, 3])/2,
                    ], axis=1)
                elif self.eval_type == 'iou':
                    gt[image_idx] = boxes

            for threshold_idx, threshold in enumerate(cls_thresholds):
                pred = {}
                for image_idx, output in enumerate(pred_raw):
                    # Get the relevant boxes (i.e., matching cls and sufficient score).
                    if len(output['boxes']) == 0:
                        boxes = output['boxes'].numpy()
                    else:
                        selector = (output['scores'] >= threshold) & (output['labels'] == cls_idx)
                        boxes = output['boxes'][selector, :].numpy()

                    # If the evaluation type is center-based, replace the predicted boxes with centers
                    # Else if it is iou-based, keep the [x1,y1,x2,y2] box format
                    if self.eval_type == 'center':
                        pred[image_idx] = np.stack([
                            (boxes[:, 0] + boxes[:, 2])/2,
                            (boxes[:, 1] + boxes[:, 3])/2,
                        ], axis=1)
                    elif self.eval_type == 'iou':
                        pred[image_idx] = boxes

                tp, fp, fn = metric_score(
                    gt, pred, self.eval_type,
                    iou_threshold=self.iou_threshold,
                    distance_tolerance=self.distance_tolerance,
                )
                self.true_positives[cls_idx][threshold_idx] += float(tp)
                self.false_positives[cls_idx][threshold_idx] += float(fp)
                self.false_negatives[cls_idx][threshold_idx] += float(fn)

    def score(self):
        best_scores = []
        best_thresholds = []

        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            best_score = None
            best_threshold = None

            if cls_idx == 0:
                best_thresholds.append(0.5)
                continue

            for threshold_idx, threshold in enumerate(cls_thresholds):
                tp = self.true_positives[cls_idx][threshold_idx]
                fp = self.false_positives[cls_idx][threshold_idx]
                fn = self.false_negatives[cls_idx][threshold_idx]

                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)

                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)

                if precision + recall < 0.001:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                if self.detail_func:
                    self.detail_func('{}_{}@{}_{}_{}'.format(self.task['name'], self.task['categories'][cls_idx], threshold, precision, recall), f1)

                if best_score is None or f1 > best_score:
                    best_score = f1
                    best_threshold = threshold

            best_scores.append(best_score)
            best_thresholds.append(best_threshold)

        # In all-background-class cases, avoid divide-by-zero errors
        if len(best_scores) == 0:
            return 0.0, best_thresholds

        return sum(best_scores) / len(best_scores), best_thresholds
