from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data
import numpy as np
import json
import cv2
import os
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian
from lib.utils.image import draw_dense_reg
import math

from lib.utils.augmentations import Augmentation

class COCO(data.Dataset):
    num_classes = 1
    default_resolution = [1024,1024]
    dense_wh = False
    reg_offset = True
    mean = np.array([0.49965, 0.49965, 0.49965],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()

        self.opt = opt
        self.img_dir0 = opt.data_dir
        self.img_dir = os.path.join(opt.data_dir, split)

        if opt.test_large_size:
            if split == 'train':
                self.resolution = [1024, 1024]
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_train.json')
            elif split == 'val':
                self.resolution = [2048, 2048]
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_val.json')
            else:  # test
                self.resolution = [2048, 2048]
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_test.json')
        else:
            self.resolution = [1024, 1024]
            if split == 'train':
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_train.json')
            elif split == 'val':
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_val.json')
            else:  # test
                self.annot_path = os.path.join(
                    self.img_dir0, 'annotations',
                    'rsipac2025_test.json')

        self.down_ratio = opt.down_ratio
        self.max_objs = opt.K
        self.seqLen = opt.seqLen

        self.class_name = [
            '__background__', 'car']
        self._valid_ids = [
            1,2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # 生成对应的category dict

        self.split = split
        self.opt = opt

        print('==> initializing RSIPAC2025 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

        if(split=='train'):
            self.aug = Augmentation()
        else:
            self.aug = None

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    # 遍历每一个标注文件解析写入detections. 输出结果使用
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                # cls_ind starts from 1, _valid_ids[0] = 1 (car category)
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, time_str):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_{}.json'.format(save_dir,time_str), 'w'))

        print('{}/results_{}.json'.format(save_dir,time_str))

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes('{}/results_{}.json'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.params.iouThrs = np.array([0.05, 0.1, 0.3, 0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        precisions = coco_eval.eval['precision']

        # 测试更低的IoU阈值
        # print("\n=== Testing with lower IoU thresholds ===")
        # for iou_thresh in [0.05]:
        #     print(f"\nEvaluating at IoU = {iou_thresh}")
        #     coco_eval_low = COCOeval(self.coco, coco_dets, "bbox")
        #     coco_eval_low.params.iouThrs = np.array([iou_thresh])
        #     coco_eval_low.evaluate()
        #     coco_eval_low.accumulate()
        #     coco_eval_low.summarize()
        #     print(f"AP at IoU={iou_thresh}: {coco_eval_low.stats[0]:.6f}")

        return stats, precisions

    def save_visual_results(self, results, save_dir, time_str, max_images=10):
        """保存带检测框的图像以检查定位准确性"""
        import cv2
        
        visual_dir = os.path.join(save_dir, f'visual_{time_str}')
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)
        
        saved_count = 0
        for image_id in list(results.keys())[:max_images]:
            # 获取图像信息
            img_info = self.coco.loadImgs(ids=[image_id])[0]
            file_name = img_info['file_name']
            
            # 构建图像路径
            if '/' in file_name:
                video_name, frame_name = file_name.split('/', 1)
                img_path = os.path.join(self.img_dir, video_name, frame_name)
            else:
                img_path = os.path.join(self.img_dir, file_name)
            
            if not os.path.exists(img_path):
                continue
                
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 绘制ground truth框
            ann_ids = self.coco.getAnnIds(imgIds=[image_id])
            anns = self.coco.loadAnns(ids=ann_ids)
            for ann in anns:
                bbox = ann['bbox']  # [x, y, w, h]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色GT框
                cv2.putText(img, 'GT', (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制预测框
            if image_id in results and 1 in results[image_id]:
                dets = results[image_id][1]  # class 1 detections
                for det in dets:
                    if det[4] >= 0.1:  # 低置信度阈值
                        x1, y1, x2, y2, conf = det
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色预测框
                        cv2.putText(img, f'P:{conf:.2f}', (int(x1), int(y2+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 保存图像
            save_path = os.path.join(visual_dir, f'img_{image_id:06d}.jpg')
            cv2.imwrite(save_path, img)
            saved_count += 1
            
        print(f"Saved {saved_count} visualization images to {visual_dir}")
        return visual_dir

    def run_eval_just(self, save_dir, time_str, iouth):
        coco_dets = self.coco.loadRes('{}/{}'.format(save_dir, time_str))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox", iouth = iouth)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_5 = coco_eval.stats
        precisions = coco_eval.eval['precision']

        return stats_5, precisions

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        seq_num = self.seqLen
        
        # 解析文件路径，适配RSIPAC2025格式
        # 文件名格式: video_name/frame_XXXXXX.jpg
        if '/' in file_name:
            video_name, frame_name = file_name.split('/', 1)
            frame_id = int(frame_name.split('_')[1].split('.')[0])
        else:
            # 兼容其他格式
            video_name = ""
            frame_name = file_name.split('.')[0].split('/')[-1]
            frame_id = int(frame_name)
        
        # 构建图像路径
        if video_name:
            img_base_path = os.path.join(self.img_dir, video_name)
        else:
            img_base_path = self.img_dir
        img_extension = '.jpg'
        
        img = np.zeros([self.resolution[0], self.resolution[1], 3, seq_num])

        for ii in range(seq_num):
            # 计算序列中的帧号
            target_frame_id = max(frame_id - ii, 1)
            
            if video_name:
                # RSIPAC2025格式: video_name/frame_XXXXXX.jpg
                frame_filename = f"frame_{target_frame_id:06d}{img_extension}"
                img_path = os.path.join(img_base_path, frame_filename)
            else:
                # 兼容原格式
                frame_filename = f"{target_frame_id:06d}{img_extension}"
                img_path = os.path.join(img_base_path, frame_filename)
            
            # 读取图像
            if os.path.exists(img_path):
                im = cv2.imread(img_path)
                if im is None:
                    print(f"Warning: Failed to load image {img_path}")
                    # 如果读取失败，使用零填充
                    im = np.zeros([self.resolution[0], self.resolution[1], 3], dtype=np.uint8)
            else:
                print(f"Warning: Image not found {img_path}")
                # 如果文件不存在，使用零填充
                im = np.zeros([self.resolution[0], self.resolution[1], 3], dtype=np.uint8)
            
            # 调整图像大小
            if im.shape[:2] != (self.resolution[0], self.resolution[1]):
                im = cv2.resize(im, (self.resolution[1], self.resolution[0]))
            
            if(ii==0):
                imgOri = im.copy()
            
            # 归一化
            inp_i = (im.astype(np.float32) / 255.)
            inp_i = (inp_i - self.mean) / self.std
            img[:,:,:,ii] = inp_i

        bbox_tol = []
        cls_id_tol = []

        for k in range(num_objs):
            ann = anns[k]
            bbox_tol.append(self._coco_box_to_bbox(ann['bbox']))
            cls_id_tol.append(self.cat_ids[ann['category_id']])

        if self.aug is not None and num_objs>0:
            bbox_tol = np.array(bbox_tol)
            cls_id_tol = np.array(cls_id_tol)
            img, bbox_tol, cls_id_tol = self.aug(img, bbox_tol, cls_id_tol)
            bbox_tol = bbox_tol.tolist()
            cls_id_tol = cls_id_tol.tolist()
            num_objs = len(bbox_tol)

        #transpose
        inp = img.transpose(2, 3, 0, 1).astype(np.float32)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0

        output_h = height // self.down_ratio
        output_w = width // self.down_ratio

        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            bbox = bbox_tol[k]
            cls_id = cls_id_tol[k]
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct[0] = np.clip(ct[0], 0, output_w - 1)
                ct[1] = np.clip(ct[1], 0, output_h - 1)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        for _ in range(num_objs, self.max_objs):
            bbox_tol.append([])


        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'imgOri': imgOri}

        if self.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']

        if self.reg_offset:
            ret.update({'reg': reg})

        ret['file_name'] = file_name

        return img_id, ret
