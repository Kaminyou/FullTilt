# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import random
import typing as t
from argparse import Namespace
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import PIL
import torch
import torchvision
from scipy.ndimage import convolve

import fulltilt.datasets.transforms as T
from fulltilt.datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
from fulltilt.datasets.prompt_util import (check_prompt_valid,
                                           prepare_prompts_aggressive,
                                           shuffle_data)
from fulltilt.util.geometry import rotate_coordinate


def generate_spotlight_cryo_tilts(
    tilt_angles: t.List[int] = [-30, 0, 30],
    image_shape: t.Tuple[int, int] = (512, 512),
    volume_height: int = 256,
    num_particles: int = 600,
    particle_radius: int = 5,
    particle_opacity: float = 0.15,
    z_slab_thickness: int = 80,
    noise_level: float = 0.08,
    spotlight_intensity: float = 0.4,
    side_artifact_max_width: int = 50,  # We will treat this area as "out of bounds" too
    return_three_channels: bool = True,
) -> t.Tuple[npt.NDArray[int], npt.NDArray[int]]:

    if z_slab_thickness > volume_height:
        raise ValueError(
            f'z_slab_thickness={z_slab_thickness} should <= volume_height={volume_height}'
        )

    h, w = image_shape
    center_x, center_y = w / 2.0, h / 2.0
    center_z = volume_height / 2.0

    # --- 1. Generate Candidates ---
    # Generate MORE than num_particles initially because we will filter some out
    initial_count = int(num_particles * 1.5)

    gx = np.random.uniform(-w / 2, w / 2, initial_count)
    gy = np.random.uniform(-h / 2, h / 2, initial_count)
    gz = np.random.uniform(-z_slab_thickness / 2, z_slab_thickness / 2, initial_count)

    # --- 2. PRE-FILTER: Keep only particles visible in ALL tilts ---
    # We assume a particle is valid initially (True)
    keep_mask = np.ones(initial_count, dtype=bool)

    # print("Checking particle visibility across all tilts...")

    for angle in tilt_angles:
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Project candidate
        x_proj = gx * cos_t + gz * sin_t
        y_proj = gy

        x_img = x_proj + center_x
        y_img = y_proj + center_y

        # Check if it falls inside the SAFE area (between the black bars)
        # Margin: We also subtract particle_radius so the *whole* ball is visible
        margin = particle_radius + 3  # MAGIC NUM

        is_visible = (
            (x_img >= margin)
            & (x_img < (w - margin))
            & (y_img >= particle_radius)
            & (y_img < (h - particle_radius))
        )

        # Determine intersection: Must be visible in current angle AND previous angles
        keep_mask = keep_mask & is_visible

    # Apply the mask
    gx = gx[keep_mask]
    gy = gy[keep_mask]
    gz = gz[keep_mask]

    # Cap at requested num_particles if we have too many survivors
    if len(gx) > num_particles:
        gx, gy, gz = gx[:num_particles], gy[:num_particles], gz[:num_particles]

    # print(f"Particles retained: {len(gx)} (Requested: {num_particles})")

    ground_truth_3d = np.stack([gx, gy, gz], axis=1)

    # --- 3. Setup Visualization Assets ---
    # Hard Disk Kernel
    y_k, x_k = np.ogrid[-particle_radius:particle_radius + 1, -particle_radius:particle_radius + 1]
    disk_kernel = (x_k**2 + y_k**2 <= particle_radius**2).astype(float)

    # Spotlight Background
    Y_grid, X_grid = np.indices(image_shape)
    spot_x = np.random.normal(center_x, w / 4)
    spot_y = np.random.normal(center_y, h / 4)
    radius_map = np.sqrt((X_grid - spot_x)**2 + (Y_grid - spot_y)**2)
    max_dist = np.sqrt(w**2 + h**2)
    base_background = 0.8 - ((radius_map / (max_dist * 0.5)) * spotlight_intensity)
    base_background = np.clip(base_background, 0.2, 1.0)

    # --- 4. Generate Images ---
    tilt_stack = []
    random_side_artifact_color = np.random.uniform(0, 0.3)
    for angle in tilt_angles:
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x_proj = gx * cos_t + gz * sin_t
        y_proj = gy
        x_img = x_proj + center_x
        y_img = y_proj + center_y

        # Place Particles
        centers_map, _, _ = np.histogram2d(
            y_img, x_img,
            bins=[h, w], range=[[0, h], [0, w]]
        )
        centers_map = (centers_map > 0).astype(float)

        particle_layer = convolve(centers_map, disk_kernel, mode='constant', cval=0.0)
        particle_layer = np.clip(particle_layer, 0, 1) * particle_opacity

        clean_image = base_background - particle_layer
        noise = np.random.normal(0, noise_level, image_shape)
        final_image = clean_image + noise

        if side_artifact_max_width > 0:
            side_artifact_width = np.random.randint(0, side_artifact_max_width + 1)
            left_side_artifact_width = np.random.randint(0, side_artifact_width + 1)
            right_side_artifact_width = side_artifact_width - left_side_artifact_width
            if left_side_artifact_width > 0:
                final_image[:, :left_side_artifact_width] = random_side_artifact_color

            if right_side_artifact_width > 0:
                final_image[:, -right_side_artifact_width:] = random_side_artifact_color

        tilt_stack.append(np.clip(final_image, 0, 1))

    # image recover to 0-255
    tilt_stack = np.array(tilt_stack)
    tilt_stack = (
        (tilt_stack - tilt_stack.min()) / (tilt_stack.max() - tilt_stack.min()) * 255
    ).astype(np.uint8)

    # shift ground_truth_3d
    ground_truth_3d += np.array([center_x, center_y, center_z])
    ground_truth_3d = ground_truth_3d.astype(int)

    if return_three_channels:
        tilt_stack = np.repeat(tilt_stack[..., np.newaxis], 3, axis=-1)

    return np.array(tilt_stack), ground_truth_3d


def generate_coco_targets(
    tilt_angles: t.List[int],
    ground_truth_3d: npt.NDArray[int],
    particle_radius: int,
    W: int,
    D: int,
):
    targets = []
    particle_diameter = particle_radius * 2
    for tilt_angle in tilt_angles:
        gt_rot_3d_points = rotate_coordinate(ground_truth_3d, tilt_angle, W=W, D=D)

        sub_targets = []
        for point_idx, gt_rot_3d_point in enumerate(gt_rot_3d_points):
            x, y, _ = gt_rot_3d_point.tolist()
            bbox = [x - particle_radius, y - particle_radius, particle_diameter, particle_diameter]
            sub_targets.append(
                {
                    'id': point_idx + 1,
                    'image_id': 1,
                    'category_id': 1,
                    'bbox': bbox,
                    'area': particle_diameter * particle_diameter,
                    'iscrowd': 0,
                    'segmentation': [],
                }
            )
        targets.append(sub_targets)
    return targets


def make_gt3d(
    ground_truth_3d: npt.NDArray[int],
    particle_radius: int,
    W: int,
    H: int,
    D: int,
):
    num = len(ground_truth_3d)
    particle_diameter = particle_radius * 2
    return {
        '3d_positions': ground_truth_3d.tolist(),
        'tomogram_size': [W, H, D],
        'particle_sizes': [particle_diameter] * num,
        'categories': ['sim'] * num,
        'labels': [0] * num,
    }


class CocoCryoRandomGenerationDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        training_generation_num: int,
        transforms: T.Compose,
        random_order_aug: bool = False,
        return_masks: bool = False,
        max_pick_prompt_num: int = 5,
        partial_pick_label: int = -1,
        drop_tilt_prompt_max_ratio: float = 0.0,
        fix_hwd: t.Optional[t.Tuple[int, int, int]] = None,  # for profiling
        fix_num_particles: t.Optional[int] = None,  # for profiling
        fix_tilt_angles: t.Optional[t.List[int]] = None,  # for profiling
    ):
        self.training_generation_num = training_generation_num
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.random_order_aug = random_order_aug

        self.max_pick_prompt_num = max_pick_prompt_num
        self.partial_pick_label = partial_pick_label
        self.drop_tilt_prompt_max_ratio = drop_tilt_prompt_max_ratio

        self.tilt_angles_choices = [
            list(range(-45, 46, 3)),
            list(range(-60, 61, 4)),
            list(range(-60, 61, 3)),
        ]

        self.fix_hwd = fix_hwd
        self.fix_num_particles = fix_num_particles
        self.fix_tilt_angles = fix_tilt_angles

    def _process_image(self, image) -> PIL.Image.Image:
        image = T.normalize_np_img(image)
        return PIL.Image.fromarray(image, 'RGB')

    def __len__(self):
        return self.training_generation_num

    def __getitem__(self, idx):
        if self.fix_tilt_angles is not None:
            tilt_angles = self.fix_tilt_angles
        else:
            tilt_angles = random.choice(self.tilt_angles_choices)

        particle_radius = random.randint(3, 8)

        if self.fix_hwd is not None:
            H, W, D = self.fix_hwd
        else:
            H = random.randint(300, 512)
            W = random.randint(300, 512)
            D = random.randint(120, 300)

        z_slab_thickness = random.randint(80, 120)
        if self.fix_num_particles is not None:
            num_particles = self.fix_num_particles
        else:
            num_particles = random.randint(100, 600)

        particle_opacity = random.uniform(0.03, 0.10)  # 0.04; 0.03-0.10
        noise_level = random.uniform(0.01, 0.10)  # 0.02; 0.01 - 0.10

        # 0.4; 0.1-0.4; High value => strong vignetting/spotlight
        spotlight_intensity = random.uniform(0.1, 0.4)
        side_artifact_max_width = random.uniform(0, 100)  # 100; 0-100

        tiltseries, ground_truth_3d = generate_spotlight_cryo_tilts(
            tilt_angles=tilt_angles,
            image_shape=(H, W),
            volume_height=D,
            z_slab_thickness=z_slab_thickness,
            num_particles=num_particles,
            particle_radius=particle_radius,
            particle_opacity=particle_opacity,
            noise_level=noise_level,
            spotlight_intensity=spotlight_intensity,
            side_artifact_max_width=side_artifact_max_width,
        )

        coco_targets = generate_coco_targets(
            tilt_angles, ground_truth_3d, particle_radius=particle_radius, W=W, D=D,
        )

        gt_3d = make_gt3d(ground_truth_3d, particle_radius, W, H, D)

        tiltseries = [self._process_image(image) for image in tiltseries]

        imgs = []
        tilts = []
        targets = []

        labels2indices_list = []
        labels2count_list = []

        resize_ratio = random.uniform(0.7, 1.0)
        pad_ratio = random.uniform(0.0, 0.05)
        # HACK: prevent OOM
        if resize_ratio * (1 + pad_ratio * 2) >= 1.0:
            pad_ratio = 0.0

        resizer = T.FixedRatioResize(resize_ratio=resize_ratio, to_resize=self.random_order_aug)
        padder = T.FixedSymmetricPad(pad_ratio=pad_ratio, to_pad=self.random_order_aug)
        flipper = T.FixedSymmetricFlip(to_flip=self.random_order_aug)

        current_index = 0
        for ordered_img_idx, (img, target, tilt) in enumerate(
            zip(tiltseries, coco_targets, tilt_angles)
        ):

            sub_labels2indices = defaultdict(list)
            sub_labels2count = defaultdict(int)

            for sub_idx, _target in enumerate(target):
                sub_labels2indices[_target['category_id']].append(sub_idx)
                sub_labels2count[_target['category_id']] += 1

            current_index += len(target)

            target = {'image_id': ordered_img_idx, 'annotations': target}
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
            max_training_tilts=21,
            min_training_tilts=15,
        )

        if not check_prompt_valid(prompts):
            raise ValueError('Prompts are all empty lists.')

        tilts = torch.as_tensor(tilts, dtype=torch.float32)

        # prepare 3d gt
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


def build_coco_random_training(
    image_set: str,
    args: Namespace,
    datasetinfo: t.Dict[str, str],
):
    training_generation_num = datasetinfo['training_generation_num']
    strong_aug = getattr(args, 'strong_aug', False)
    print('build_coco_random_training!')

    random_order_aug = getattr(args, 'random_order_aug', False)
    max_pick_prompt_num = getattr(args, 'max_pick_prompt_num', 5)
    partial_pick_label = getattr(args, 'partial_pick_label', -1)
    drop_tilt_prompt_max_ratio = getattr(args, 'drop_tilt_prompt_max_ratio', 0.0)

    fix_hwd = None
    fix_num_particles = None
    fix_tilt_angles = None
    # TODO: handle them elegantly
    if image_set == 'train' and not random_order_aug:
        raise NotImplementedError(
            'Currently we only support random_order_aug in training set.'
            'Please set random_order_aug to True.'
        )

    if image_set != 'train':
        random_order_aug = False
        max_pick_prompt_num = 5
        partial_pick_label = -1
        drop_tilt_prompt_max_ratio = 0.0

        # NOTE: used for profiling only
        # so image_set != train but still use this module
        fix_hwd = (256, 256, 256)
        fix_num_particles = 100
        fix_tilt_angles = list(range(-45, 46, 3))

    dataset = CocoCryoRandomGenerationDetection(
        training_generation_num=training_generation_num,
        transforms=make_coco_transforms(
            image_set,
            strong_aug=strong_aug,
        ),
        random_order_aug=random_order_aug,
        max_pick_prompt_num=max_pick_prompt_num,
        partial_pick_label=partial_pick_label,
        drop_tilt_prompt_max_ratio=drop_tilt_prompt_max_ratio,
        fix_hwd=fix_hwd,
        fix_num_particles=fix_num_particles,
        fix_tilt_angles=fix_tilt_angles,
    )

    return dataset
