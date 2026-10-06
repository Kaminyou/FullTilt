# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from
https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import os
import random
import typing as t
from argparse import Namespace
from collections import defaultdict

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import fulltilt.datasets.transforms as T
from fulltilt.datasets.prompt_util import (check_prompt_valid,
                                           prepare_prompts_aggressive,
                                           shuffle_data)

__all__ = ['build']


class CocoCryoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        group_file: str,
        group_3d_file: str,
        transforms: T.Compose,
        random_order_aug: bool = False,
        return_masks: bool = False,
        max_pick_prompt_num: int = 5,
        partial_pick_label: int = -1,
        drop_tilt_prompt_max_ratio: float = 0.0,
    ):
        super(CocoCryoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.group_file = group_file
        self.group_3d_file = group_3d_file
        self.random_order_aug = random_order_aug

        self.max_pick_prompt_num = max_pick_prompt_num
        self.partial_pick_label = partial_pick_label
        self.drop_tilt_prompt_max_ratio = drop_tilt_prompt_max_ratio

        self._load_group_file()
        self._load_group_3d_file()
        self._get_idx_rev_map()

    def _load_group_file(self):
        with open(self.group_file, 'r') as f:
            groups = json.load(f)
        self.group_id2img_ids = groups  # {str->List[int]}
        # NOTE: to make thing stable
        self.group_keys = sorted(list(self.group_id2img_ids.keys()))

    def _load_group_3d_file(self):
        with open(self.group_3d_file, 'r') as f:
            group_3dgt = json.load(f)
        self.group_id23dgt = group_3dgt  # {str->dict}
        # dict spec:
        # '3d_positions' -> list[list[int]] (x, y, z)
        # 'tomogram_size' -> list[int] (W, H, D)
        # 'particle_sizes' -> list[int]
        # 'categories' -> list[str]
        # 'labels' -> list[int]

    def _load_image(self, id: int) -> PIL.Image.Image:
        meta = self.coco.loadImgs(id)[0]
        path = meta['file_name']
        abs_path = os.path.join(self.root, path)

        image = PIL.Image.open(abs_path).convert('RGB')
        image = np.asarray(image)
        image = T.normalize_np_img(image)

        return PIL.Image.fromarray(image, 'RGB')

    def __len__(self):
        return len(self.group_id2img_ids)

    def _get_idx_rev_map(self):
        self.ids_rev_map = {}
        for i, img_id in enumerate(self.ids):
            self.ids_rev_map[img_id] = i

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data.
        """
        group_key = self.group_keys[idx]
        img_ids = self.group_id2img_ids[group_key]  # img_ids start from zero; coco starts from 1
        imgs = []
        tilts = []
        targets = []

        labels2indices_list = []
        labels2count_list = []

        resize_ratio = random.uniform(0.7, 1.0)
        pad_ratio = random.uniform(0.0, 0.05)
        if resize_ratio * (1 + pad_ratio * 2) >= 1.0:  # NOTE: address OOM
            pad_ratio = 0.0
        resizer = T.FixedRatioResize(resize_ratio=resize_ratio, to_resize=self.random_order_aug)
        padder = T.FixedSymmetricPad(pad_ratio=pad_ratio, to_pad=self.random_order_aug)
        flipper = T.FixedSymmetricFlip(to_flip=self.random_order_aug)

        current_index = 0
        for img_id in img_ids:
            coco_img_id = self.ids[img_id]  # get coco img_id
            ori_img_id = self.ids_rev_map[coco_img_id]
            img, target = super(CocoCryoDetection, self).__getitem__(ori_img_id)

            sub_labels2indices = defaultdict(list)
            sub_labels2count = defaultdict(int)

            for sub_idx, _target in enumerate(target):
                sub_labels2indices[_target['category_id']].append(sub_idx)
                sub_labels2count[_target['category_id']] += 1

            current_index += len(target)
            tilt = self.coco.loadImgs(coco_img_id)[0]['tilt']

            target = {'image_id': coco_img_id, 'annotations': target}
            img, target = self.prepare(img, target)

            if self._transforms is not None:
                img, target = resizer(img, target)
                img, target = padder(img, target)
                img, target = flipper(img, target)
                img, target = self._transforms(img, target)

            imgs.append(img)
            tilts.append(tilt)
            targets.append(target)

            labels2indices_list.append(sub_labels2indices)
            labels2count_list.append(sub_labels2count)

        # NOTE: tricky
        # shuffle first as some tilts will be discarded
        # then we drop tile's prompts
        # if do it in a reverse order, the tilts's that have
        # prompts might totally discarded -> nan in decoder
        prompts, targets, keep = prepare_prompts_aggressive(
            targets=targets,
            labels2indices_list=labels2indices_list,
            labels2count_list=labels2count_list,
            max_pick_prompt_num=self.max_pick_prompt_num,
            partial_pick_label=self.partial_pick_label,
            drop_tilt_prompt_max_ratio=self.drop_tilt_prompt_max_ratio,
        )

        imgs, tilts, targets, prompts = shuffle_data(
            imgs,
            tilts,
            targets,
            prompts,
            random_order_aug=self.random_order_aug,
        )

        if not check_prompt_valid(prompts):
            raise ValueError('Prompts are all empty lists.')

        tilts = torch.as_tensor(tilts, dtype=torch.float32)

        # prepare 3d gt
        gt_3d = self.group_id23dgt[group_key]
        # 1. Load Data
        positions_3d = np.array(gt_3d['3d_positions']).astype(float)[keep]  # [N, 3] (x, y, z)
        tomogram_size = np.array(gt_3d['tomogram_size']).astype(float)  # [3] (W, H, D)
        particle_sizes = np.array(gt_3d['particle_sizes']).astype(float)[keep]  # [N] or [N, 1]
        labels = np.array(gt_3d['labels']).astype(int)[keep]  # [N]

        # TODO: tricky
        # ---------------
        # coco has a + 1 label
        # ---------------
        labels = labels + 1
        if not np.array_equal(labels, targets[0]['labels']):  # check if match
            raise ValueError('Group labels do not match tilt labels.')

        # 2. Normalize Positions (0~1)
        # positions_3d / [W, H, D]
        positions_3d_normalized = positions_3d / tomogram_size[None, :]

        # 3. Normalize Sizes (0~1) & Expand to (w, h)
        # particle_sizes is usually diameter. We assume isotropic in x-y plane.
        # w_norm = size / W, h_norm = size / H
        if particle_sizes.ndim == 1:
            particle_sizes = particle_sizes[:, None]  # [N, 1]

        # W, H for normalization
        size_norm_factor = tomogram_size[:2]  # [W, H]

        # Repeat size for w and h: [N, 1] -> [N, 2]
        particle_wh = np.repeat(particle_sizes, 2, axis=1)  # [N, 2]

        particle_wh_normalized = particle_wh / size_norm_factor[None, :]

        # 4. Combine into [N, 5] (cx, cy, cz, w, h)
        boxes_3d = np.concatenate([positions_3d_normalized, particle_wh_normalized], axis=1)

        gt_3d_torch = {
            'boxes_3d': torch.as_tensor(boxes_3d, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'tomogram_size': torch.as_tensor(tomogram_size, dtype=torch.float32),
        }
        gt_3d_torch = resizer.transform_3dgt(gt_3d_torch)
        gt_3d_torch = padder.transform_3dgt(gt_3d_torch)
        gt_3d_torch = flipper.transform_3dgt(gt_3d_torch)

        return imgs, tilts, targets, prompts, gt_3d_torch

    def getitem_pure(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data.
        """
        group_key = self.group_keys[idx]
        img_ids = self.group_id2img_ids[group_key]  # img_ids start from zero; coco starts from 1
        imgs = []
        tilts = []
        targets = []

        labels2indices_list = []
        labels2count_list = []

        for img_id in img_ids:
            coco_img_id = self.ids[img_id]  # get coco img_id
            ori_img_id = self.ids_rev_map[coco_img_id]
            img, target = super(CocoCryoDetection, self).__getitem__(ori_img_id)

            sub_labels2indices = defaultdict(list)
            sub_labels2count = defaultdict(int)

            for sub_idx, _target in enumerate(target):
                sub_labels2indices[_target['category_id']].append(sub_idx)
                sub_labels2count[_target['category_id']] += 1

            tilt = self.coco.loadImgs(coco_img_id)[0]['tilt']

            target = {'image_id': coco_img_id, 'annotations': target}
            img, target = self.prepare(img, target)

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            imgs.append(img)
            tilts.append(tilt)
            targets.append(target)

            labels2indices_list.append(sub_labels2indices)
            labels2count_list.append(sub_labels2count)

        tilts = torch.as_tensor(tilts, dtype=torch.float32)

        # prepare 3d gt
        gt_3d = self.group_id23dgt[group_key]
        # 1. Load Data
        positions_3d = np.array(gt_3d['3d_positions']).astype(float)  # [N, 3] (x, y, z)
        tomogram_size = np.array(gt_3d['tomogram_size']).astype(float)  # [3] (W, H, D)
        particle_sizes = np.array(gt_3d['particle_sizes']).astype(float)  # [N] or [N, 1]
        labels = np.array(gt_3d['labels']).astype(int)  # [N]

        # TODO: tricky
        # ---------------
        # coco has a + 1 label
        # ---------------
        labels = labels + 1
        if not np.array_equal(labels, targets[0]['labels']):  # check if match
            raise ValueError(
                'Group labels do not match tilt labels.'
                f'Group label ({labels.shape}): {labels}'
                f'Tilt label ({targets[0]["labels"].shape}): {targets[0]["labels"]}'
            )

        positions_3d_normalized = positions_3d / tomogram_size[None, :]
        if particle_sizes.ndim == 1:
            particle_sizes = particle_sizes[:, None]  # [N, 1]

        size_norm_factor = tomogram_size[:2]  # [W, H]
        particle_wh = np.repeat(particle_sizes, 2, axis=1)  # [N, 2]

        particle_wh_normalized = particle_wh / size_norm_factor[None, :]
        boxes_3d = np.concatenate([positions_3d_normalized, particle_wh_normalized], axis=1)

        gt_3d_torch = {
            'boxes_3d': torch.as_tensor(boxes_3d, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'tomogram_size': torch.as_tensor(tomogram_size, dtype=torch.float32),
        }

        return imgs, tilts, targets, gt_3d_torch


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj['segmentation'] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # TODO
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # keep = np.ones(len(boxes), dtype=bool)
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        if self.return_masks:
            target['masks'] = masks
        target['image_id'] = image_id
        if keypoints is not None:
            target['keypoints'] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in anno])
        target['area'] = area[keep]
        target['iscrowd'] = iscrowd[keep]

        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(
    image_set: str,
    strong_aug: bool = False,
):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        if strong_aug:
            return T.Compose([
                T.RandomGaussianNoise(p=0.2, mean=0.0, std=0.1),
                T.RandomGrayRectangle(p=0.2, max_size=150),
                normalize,
            ])
        else:
            return T.Compose([
                normalize,
            ])

    if image_set in ['val', 'test']:
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(
    image_set: str,
    args: Namespace,
    datasetinfo: t.Dict[str, str],
) -> CocoCryoDetection:

    img_folder = datasetinfo['root']
    ann_file = datasetinfo['anno']

    group_file = datasetinfo.get('group_file', None)
    group_3d_file = datasetinfo.get('group_3d_file', None)

    if group_file is None:
        raise ValueError('group_file should be provided.')

    if group_3d_file is None:
        raise ValueError('group_3d_file should be provided.')
    # print(img_folder, ann_file, group_file)

    strong_aug = getattr(args, 'strong_aug', False)
    random_order_aug = getattr(args, 'random_order_aug', False)
    max_pick_prompt_num = getattr(args, 'max_pick_prompt_num', 5)
    partial_pick_label = getattr(args, 'partial_pick_label', -1)
    drop_tilt_prompt_max_ratio = getattr(args, 'drop_tilt_prompt_max_ratio', 0.0)

    # TODO: handle them elegantly
    if image_set == 'train' and not random_order_aug:
        raise NotImplementedError(
            'Currently we only support random_order_aug in training set.'
            'Please set random_order_aug to True.'
        )

    if image_set != 'train':
        strong_aug = False
        random_order_aug = False
        max_pick_prompt_num = 5
        partial_pick_label = -1
        drop_tilt_prompt_max_ratio = 0.0

    dataset = CocoCryoDetection(
        img_folder,
        ann_file,
        group_file=group_file,
        group_3d_file=group_3d_file,
        transforms=make_coco_transforms(
            image_set,
            strong_aug=strong_aug,
        ),
        random_order_aug=random_order_aug,
        max_pick_prompt_num=max_pick_prompt_num,
        partial_pick_label=partial_pick_label,
        drop_tilt_prompt_max_ratio=drop_tilt_prompt_max_ratio,
    )
    return dataset
