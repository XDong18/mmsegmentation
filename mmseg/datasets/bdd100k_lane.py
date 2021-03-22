from mmcv.runner import get_dist_info
import os.path as osp
import random
import mmcv
import numpy as np
from .custom import CustomDataset
# from mmcv.parallel import DataContainer as DC
# from pycocotools.bdd import BDD
# from .utils import to_tensor, random_scale, random_crop
from .builder import DATASETS
from .custom import CustomDataset
import os
from PIL import Image
import json
import torch
from functools import reduce
from mmseg.core import mean_iou
from mmcv.utils import print_log

LABEL_NAMES =['lane_dir', 'lane_sty', 'lane_type']

@DATASETS.register_module()
class Bdd100k_LaneDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = None

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, split, **kwargs):
        super(Bdd100k_LaneDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir)
    
    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps_0 = []
        gt_seg_maps_1 = []
        gt_seg_maps_2 = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps_0.append(gt_seg_map[:,:,0])
            gt_seg_maps_1.append(gt_seg_map[:,:,1])
            gt_seg_maps_2.append(gt_seg_map[:,:,2])
            rank, world_size = get_dist_info()
            if rank == 0:
                print('\ndataset', gt_seg_map[:,:,0].max(), gt_seg_map[:,:,1].max(), gt_seg_map[:,:,2].max(),'\ndataset')
        return gt_seg_maps_0, gt_seg_maps_1, gt_seg_maps_2
    
    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[list]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps_0, gt_seg_maps_1, gt_seg_maps_2 = self.get_gt_seg_maps()
        results_0 = results[0]
        results_1 = results[1]
        results_2 = results[2]
        if self.CLASSES is None:
            num_classes_0 = 3
            num_classes_1 = 3
            num_classes_2 = 9
        else:
            # num_classes = len(self.CLASSES) # TODO fix
            pass
        num_classes = [3, 3, 9]

        all_acc_0, acc_0, iou_0 = mean_iou(
            results_0, gt_seg_maps_0, num_classes_0, ignore_index=self.ignore_index)
        all_acc_1, acc_1, iou_1 = mean_iou(
            results_1, gt_seg_maps_1, num_classes_1, ignore_index=self.ignore_index)
        all_acc_2, acc_2, iou_2 = mean_iou(
            results_2, gt_seg_maps_2, num_classes_2, ignore_index=self.ignore_index)
        all_acc = [all_acc_0, all_acc_1, all_acc_2]
        acc = [acc_0, acc_1, acc_2]
        iou = [iou_0, iou_1, iou_2]

        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = [tuple(range(num_classes_i)) for num_classes_i in num_classes]
        else:
            # class_names = self.CLASSES #TODO fix
            pass
        for idx in range(3):
            for i in range(num_classes[idx]):
                iou_str = '{:.2f}'.format(iou[idx][i] * 100)
                acc_str = '{:.2f}'.format(acc[idx][i] * 100)
                summary_str += line_format.format(LABEL_NAMES[idx] + '_' + str(class_names[idx][i]), iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

        for idx in range(3):
            iou_str = '{:.2f}'.format(np.nanmean([idx]) * 100)
            acc_str = '{:.2f}'.format(np.nanmean(acc[idx]) * 100)
            all_acc_str = '{:.2f}'.format(all_acc[idx] * 100)
            summary_str += line_format.format(LABEL_NAMES[idx], iou_str, acc_str,
                                            all_acc_str)
        print_log(summary_str, logger)

        for idx in range(3):
            eval_results[LABEL_NAMES[idx] + '_' + 'mIoU'] = np.nanmean(iou[idx])
            eval_results[LABEL_NAMES[idx] + '_' + 'mAcc'] = np.nanmean(acc[idx])
            eval_results[LABEL_NAMES[idx] + '_' + 'aAcc'] = all_acc[idx]

        return eval_results
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos[:100]
