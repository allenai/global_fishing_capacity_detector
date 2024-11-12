import argparse
import importlib
import math
import os

import numpy as np
import random
import tqdm
import torch

from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmrotate.core import poly2obb_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply the GFCD vessel detection model on val set",
    )
    parser.add_argument(
        "--command",
        type=str,
        help="One of 'score', 'bucket_score', or 'vis'",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="The path to the vessel detection dataset",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="The experiment directory to load weights from",
        default="gfcd_detector_exp_dir",
    )
    parser.add_argument(
        "--train_module",
        type=str,
        help="The module used to train the model to load config from",
        default="detector_mmrotate.gfcd_train",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Only apply on images with this prefix (optional)",
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory (for vis command)",
        default=None,
    )
    args = parser.parse_args()

    # Load the model configuration.
    gfcd_train = importlib.import_module(args.train_module)
    cfg = gfcd_train.get_cfg()

    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    device = "cuda:0"
    checkpoint = load_checkpoint(model, os.path.join(args.exp_dir, "latest.pth"), map_location=device)
    model.cfg = cfg
    model.to(device)
    model.eval()

    image_dir = os.path.join(args.data_root, "val", "images")
    label_dir = os.path.join(args.data_root, "val", "labelTxt")
    fnames = os.listdir(image_dir)

    if args.prefix:
        fnames = [fname for fname in fnames if fname.startswith(args.prefix)]

    # Scoring.
    def do_scoring():
        from gfcd_eval_lib import DetectF1Evaluator
        evaluator = DetectF1Evaluator(task={
            "name": "minderoo",
            "categories": ["unknown", "vessel"],
        }, spec={"EvalDistanceTolerance": 20}, detail_func=print)
        for fname in tqdm.tqdm(fnames):
            img_path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))

            # Get annotations.
            gt_boxes = []
            gt_labels = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(part)) for part in parts[0:8]]
                    cx = np.mean([x1, x2, x3, x4])
                    cy = np.mean([y1, y2, y3, y4])
                    gt_boxes.append([cx-20, cy-20, cx+20, cy+20])
                    gt_labels.append(1)

            # Get predictions.
            result = inference_detector(model, img_path)
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            for box in result[0]:
                cx = box[0]
                cy = box[1]
                pred_boxes.append([cx-20, cy-20, cx+20, cy+20])
                pred_labels.append(1)
                pred_scores.append(box[5])

            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.int32)
            else:
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.tensor(gt_labels, dtype=torch.int32)
            if len(pred_boxes) == 0:
                pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_labels = torch.zeros((0,), dtype=torch.int32)
                pred_scores = torch.zeros((0,), dtype=torch.float32)
            else:
                pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
                pred_labels = torch.tensor(pred_labels, dtype=torch.int32)
                pred_scores = torch.tensor(pred_scores, dtype=torch.float32)

            evaluator.evaluate([{
                "boxes": gt_boxes,
                "labels": gt_labels,
            }], [{
                "boxes": pred_boxes,
                "labels": pred_labels,
                "scores": pred_scores,
            }])
        print(evaluator.score())

    def do_bucket_scoring(buckets, distance_threshold=1.0, check_width=False):
        """
        Report precision/recall curves in different buckets of vessel lengths.
        If you want to use buckets like [0 to 50 pixels] [50 to 100 pixels] [>100 pixels],
        then set buckets = [50, 100, 9999] (last just needs to be really large).

        Distance threshold is a multiplier on the vessel width.
        """
        thresholds = [x/20 for x in range(1, 20)]
        scores = {}
        for bucket_idx in range(len(buckets)):
            for threshold_idx in range(len(thresholds)):
                scores[(bucket_idx, threshold_idx)] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }

        def get_bucket(value):
            for bucket_idx, bucket in enumerate(buckets):
                if value < bucket:
                    return bucket_idx
            raise Exception('last bucket too small')

        for fname in tqdm.tqdm(fnames):
            img_path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))

            # Get annotations.
            gt_vessels = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(part)) for part in parts[0:8]]
                    cx = np.mean([x1, x2, x3, x4])
                    cy = np.mean([y1, y2, y3, y4])
                    dims = [
                        math.sqrt((x2-x1)**2 + (y2-y1)**2),
                        math.sqrt((x3-x2)**2 + (y3-y2)**2),
                    ]
                    length = max(dims)
                    width = min(dims)
                    gt_vessels.append((cx, cy, length, width))

            # Get predictions.
            result = inference_detector(model, img_path)
            pred_vessels = []
            for box in result[0]:
                cx = box[0]
                cy = box[1]
                length = max(box[2], box[3])
                width = min(box[2], box[3])
                score = box[5]
                pred_vessels.append((cx, cy, length, width, score))

            for threshold_idx, threshold in enumerate(thresholds):
                cur_pred = [(x, y, length, width) for x, y, length, width, score in pred_vessels if score >= threshold]

                # Match to gt.
                seen_pred_idx = set()
                for gt_x, gt_y, gt_length, gt_width in gt_vessels:
                    if check_width:
                        bucket_idx = get_bucket(gt_width)
                    else:
                        bucket_idx = get_bucket(gt_length)

                    best_pred_idx = None
                    best_distance = None
                    for pred_idx, (pred_x, pred_y, pred_length, pred_width) in enumerate(cur_pred):
                        distance = math.sqrt((pred_x-gt_x)**2 + (pred_y-gt_y)**2)
                        if distance > gt_width * distance_threshold:
                            continue
                        if best_distance is None or distance < best_distance:
                            best_pred_idx = pred_idx
                            best_distance = distance

                    if best_pred_idx is None:
                        scores[(bucket_idx, threshold_idx)]["fn"] += 1
                        continue

                    seen_pred_idx.add(best_pred_idx)
                    scores[(bucket_idx, threshold_idx)]["tp"] += 1

                # Add remaining predictions as false positives.
                for pred_idx, (pred_x, pred_y, pred_length, pred_width) in enumerate(cur_pred):
                    if pred_idx in seen_pred_idx:
                        continue

                    if check_width:
                        bucket_idx = get_bucket(pred_width)
                    else:
                        bucket_idx = get_bucket(pred_length)

                    scores[(bucket_idx, threshold_idx)]["fp"] += 1

        for bucket_idx, bucket in enumerate(buckets):
            for threshold_idx, threshold in enumerate(thresholds):
                if (bucket_idx, threshold_idx) not in scores:
                    continue
                d = scores[(bucket_idx, threshold_idx)]
                tp = d["tp"]
                fp = d["fp"]
                fn = d["fn"]
                precision = tp / (tp + fp) if tp > 0 else 0
                recall = tp / (tp + fn) if tp > 0 else 0
                print(f"{bucket}\t{threshold}\t{precision}\t{recall}")

    # Visualization
    def do_vis(vis_dir, score_thr=0.5):
        random.shuffle(fnames)
        for fname in tqdm.tqdm(fnames):
            img_path = os.path.join(image_dir, fname)
            result = inference_detector(model, img_path)
            model.show_result(img_path, result, out_file=os.path.join(vis_dir, fname), score_thr=score_thr, bbox_color=(255, 255, 0), text_color=(255, 255, 0))

            gt_boxes = []
            with open(os.path.join(label_dir, fname.replace(".png", ".txt"))) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    gt_box = np.array(parts[0:8], dtype=np.float32)
                    gt_box = poly2obb_np(gt_box, "oc")
                    gt_boxes.append(gt_box + (1,))

            if len(gt_boxes) == 0:
                gt_boxes = [np.zeros((0, 6), dtype=np.float32)]
            else:
                gt_boxes = [np.array(gt_boxes, dtype=np.float32)]

            model.show_result(img_path, gt_boxes, out_file=os.path.join(vis_dir, fname.replace(".png", "_gt.png")), score_thr=score_thr, bbox_color=(255, 255, 0), text_color=(255, 255, 0))

    if args.command == "score":
        do_scoring()
    elif args.command == "bucket_score":
        do_bucket_scoring([15, 30, 45, 90, 180, 9999], check_width=True)
    elif args.command == "vis":
        do_vis(args.out_dir)
