# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import copy
import math
import random
import typing as t

import numpy as np
import numpy.typing as npt
import PIL
import torch
import torchvision.transforms.functional as F

from fulltilt.util import box_ops, misc


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = box_ops.box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target['boxes'] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(transform)
        format_string += '\n)'
        return format_string


def unnormalize_to_numpy(tensor: torch.Tensor) -> npt.NDArray:
    """
    tensor: torch.Tensor of shape (3, H, W), normalized
    return: np.ndarray of shape (H, W, 3), RGB in [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # (3, H, W) -> (H, W, 3)
    img = tensor.permute(1, 2, 0).cpu().numpy()

    # unnormalize
    img = img * std + mean

    # clip and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def normalize_np_img(
    x: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    x = (x - x.min()) / (x.max() - x.min())
    x *= 255
    x = x.astype(np.uint8)
    return x


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target['boxes'] = scaled_boxes

    if 'area' in target:
        area = target['area']
        scaled_area = area * (ratio_width * ratio_height)
        target['area'] = scaled_area

    h, w = size
    target['size'] = torch.tensor([h, w])

    if 'masks' in target:
        target['masks'] = (
            misc.interpolate(target['masks'][:, None].float(), size, mode='nearest')[:, 0] > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target['size'] = torch.tensor(padded_image.size[::-1])
    if 'masks' in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class FixedRatioResize:
    def __init__(
        self,
        resize_ratio: t.Optional[float] = None,
        low: float = 0.7,
        high: float = 1.2,
        to_resize: bool = False,
    ):
        if not to_resize:
            self.resize_ratio = 1.0
        elif resize_ratio is not None:
            self.resize_ratio = resize_ratio
        else:
            self.resize_ratio = random.uniform(low, high)

    def __call__(self, img: PIL.Image.Image, target: t.Dict):
        if math.isclose(self.resize_ratio, 1.0):
            return img, target
        w, h = img.size
        new_w = int(w * self.resize_ratio)
        new_h = int(h * self.resize_ratio)
        return resize(img, target, (new_w, new_h))

    def transform_3dgt(self, gt_3d: t.Dict[str, torch.tensor]):
        if math.isclose(self.resize_ratio, 1.0):
            return gt_3d
        new_gt_3d = copy.deepcopy(gt_3d)
        # boxes_3d has been normalized -> no need to change
        # labels -> no need to change
        new_gt_3d['tomogram_size'] = new_gt_3d['tomogram_size'] * self.resize_ratio
        return new_gt_3d


class FixedSymmetricPad:
    def __init__(
        self,
        pad_ratio: t.Optional[float] = None,
        low: float = 0.0,
        high: float = 0.10,
        to_pad: bool = False,
    ):
        if not to_pad:
            self.pad_ratio = 0.0
        elif pad_ratio is not None:
            self.pad_ratio = pad_ratio
        else:
            self.pad_ratio = random.uniform(low, high)

        self.applied_offsets = (0, 0)  # (pad_w, pad_h)
        self.new_wh = (0, 0)

    def __call__(self, img: PIL.Image.Image, target: t.Dict):
        if math.isclose(self.pad_ratio, 0.0):
            return img, target

        w, h = img.size
        pad_w = int(w * self.pad_ratio)
        pad_h = int(h * self.pad_ratio)

        self.applied_offsets = (pad_w, pad_h)
        self.new_wh = (w + 2 * pad_w, h + 2 * pad_h)

        # Symmetric padding: (left, top, right, bottom)
        img = F.pad(img, (pad_w, pad_h, pad_w, pad_h), fill=127)

        if target is not None and 'boxes' in target:
            # target["boxes"] is [N, 4] in XYXY unnormalized format
            boxes = target['boxes']
            # Simply add the offset to xmin, ymin, xmax, ymax
            boxes[:, [0, 2]] += pad_w
            boxes[:, [1, 3]] += pad_h
            target['boxes'] = boxes

        target['size'] = torch.tensor([self.new_wh[0], self.new_wh[1]])

        return img, target

    def transform_3dgt(self, gt_3d: t.Dict[str, torch.Tensor]):
        """
        Calculates new normalized 3D coordinates based on the
        enlarged volumetric container.
        """
        if math.isclose(self.pad_ratio, 0.0):
            return gt_3d

        new_gt_3d = copy.deepcopy(gt_3d)
        # Current size (after resize, before padding): [W', H', D']
        curr_size = gt_3d['tomogram_size']

        pad_w, pad_h = self.applied_offsets
        # Pad D symmetrically as well to keep the volume isotropic
        pad_d = int(curr_size[2] * self.pad_ratio)

        new_w, new_h = self.new_wh
        new_d = curr_size[2] + 2 * pad_d

        # 1. Update Metadata
        new_gt_3d['tomogram_size'] = torch.tensor([new_w, new_h, new_d], dtype=torch.float32)

        # 2. Update 3D Boxes: [cx, cy, cz, w, h] (normalized)
        boxes_3d = new_gt_3d['boxes_3d']

        # Shift centers and re-normalize
        boxes_3d[:, 0] = (boxes_3d[:, 0] * curr_size[0] + pad_w) / new_w
        boxes_3d[:, 1] = (boxes_3d[:, 1] * curr_size[1] + pad_h) / new_h
        boxes_3d[:, 2] = (boxes_3d[:, 2] * curr_size[2] + pad_d) / new_d

        # Re-normalize particle sizes (physical size is unchanged)
        boxes_3d[:, 3] = (boxes_3d[:, 3] * curr_size[0]) / new_w
        boxes_3d[:, 4] = (boxes_3d[:, 4] * curr_size[1]) / new_h

        new_gt_3d['boxes_3d'] = boxes_3d
        return new_gt_3d


class FixedSymmetricFlip:
    def __init__(self, p: float = 0.5, to_flip: bool = False):
        """
        Args:
            p: Probability of flipping for each axis.
            to_flip: Global toggle for the transform.
        """
        self.to_flip = to_flip
        self.p = p
        self.do_hflip = False
        self.do_vflip = False
        self._initialized = False

    def __call__(self, img: PIL.Image.Image, target: t.Dict):
        if not self.to_flip:
            return img, target

        # Determine flip status once per group (on the first call)
        if not self._initialized:
            self.do_hflip = random.random() < self.p
            self.do_vflip = random.random() < self.p
            self._initialized = True

        w, h = img.size

        if self.do_hflip:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            if target is not None and 'boxes' in target:
                # target["boxes"] is [N, 4] in XYXY unnormalized pixels
                boxes = target['boxes']
                old_xmin = boxes[:, 0].clone()
                # new_xmin = width - old_xmax
                boxes[:, 0] = w - boxes[:, 2]
                boxes[:, 2] = w - old_xmin
                target['boxes'] = boxes

        if self.do_vflip:
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            if target is not None and 'boxes' in target:
                # new_ymin = height - old_ymax
                boxes = target['boxes']
                old_ymin = boxes[:, 1].clone()
                boxes[:, 1] = h - boxes[:, 3]
                boxes[:, 3] = h - old_ymin
                target['boxes'] = boxes

        return img, target

    def transform_3dgt(self, gt_3d: t.Dict[str, torch.Tensor]):
        """
        Synchronizes the 3D ground truth with the 2D flips.
        Flipping X in 2D requires flipping X and Z in 3D to maintain
        the projection geometry.
        """
        if not self.to_flip or (not self.do_hflip and not self.do_vflip):
            return gt_3d

        new_gt_3d = copy.deepcopy(gt_3d)
        # boxes_3d: [N, 5] (cx, cy, cz, w, h) normalized [0, 1]
        boxes_3d = new_gt_3d['boxes_3d']

        if self.do_hflip:
            # Mirror X coordinate
            boxes_3d[:, 0] = 1.0 - boxes_3d[:, 0]
            # Mirror Z coordinate to maintain chirality and projection consistency
            boxes_3d[:, 2] = 1.0 - boxes_3d[:, 2]

        if self.do_vflip:
            # Mirror Y coordinate
            boxes_3d[:, 1] = 1.0 - boxes_3d[:, 1]

        new_gt_3d['boxes_3d'] = boxes_3d
        return new_gt_3d


class RandomGaussianNoise:
    def __init__(self, p: float = 0.2, mean: float = 0.0, std: float = 0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        if random.random() < self.p:
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(self.mean, self.std, img_array.shape[:2])[:, :, None]
            img_array = np.clip(img_array + noise, 0, 1)
            img = PIL.Image.fromarray((img_array * 255).astype(np.uint8))
        return img, target


class RandomGrayRectangle:
    def __init__(self, p: float = 0.2, max_size: int = 30):
        self.p = p
        self.max_size = max_size

    def __call__(self, img, target):
        if random.random() < self.p:
            w, h = img.size
            rw, rh = random.randint(10, self.max_size), random.randint(10, self.max_size)
            rx, ry = random.randint(0, w - rw), random.randint(0, h - rh)
            img_np = np.array(img)
            img_np[ry:ry + rh, rx:rx + rw, :] = 128  # Gray
            img = PIL.Image.fromarray(img_np)
        return img, target
