import os.path
import sys

import mmcv
from mmcv import Config
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.apis import train_detector


@ROTATED_DATASETS.register_module()
class TinyDataset(DOTADataset):
    CLASSES = ('vessel',)

    def load_annotations(self, ann_folder):
        data_infos = super(TinyDataset, self).load_annotations(ann_folder)
        for info in data_infos:
            info["height"] = 1024
            info["width"] = 1024
        return data_infos


def get_cfg():
    data_root = sys.argv[1]
    cfg = Config.fromfile('detector_mmrotate/configs/roi_trans/train_orig_31.py')
    cfg.data_root = data_root

    cfg.data.train.type = 'TinyDataset'
    cfg.data.train.data_root = os.path.join(data_root, "train")
    cfg.data.train.ann_file = 'labelTxt/'
    cfg.data.train.img_prefix = 'images/'

    cfg.data.val.type = 'TinyDataset'
    cfg.data.val.data_root = os.path.join(data_root, "val")
    cfg.data.val.ann_file = 'labelTxt/'
    cfg.data.val.img_prefix = 'images/'

    cfg.data.test.type = 'TinyDataset'
    cfg.data.test.data_root = os.path.join(data_root, "val")
    cfg.data.test.ann_file = 'labelTxt/'
    cfg.data.test.img_prefix = 'images/'

    cfg.model.roi_head.bbox_head[0].num_classes = 1
    cfg.model.roi_head.bbox_head[1].num_classes = 1

    # Download from here:
    # https://github.com/open-mmlab/mmrotate/blob/main/configs/roi_trans/README.md
    cfg.load_from = 'roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth'

    cfg.work_dir = 'gfcd_detector_exp_dir/'
    cfg.optimizer.lr = 1e-4
    cfg.runner.max_epochs = 120
    cfg.evaluation.metric = 'mAP'
    cfg.seed = 0
    cfg.gpu_ids = range(1)
    cfg.device='cuda'

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
    ]
    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
