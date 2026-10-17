# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import copy
import typing as t

import torch


def prepare_input_data(
    samples: torch.Tensor,
    tilts: t.List[torch.Tensor],
    targets: t.List[t.List[t.Dict[str, t.Any]]],
    prompts: t.List[t.List[t.Dict[str, torch.Tensor]]],
    gt_3d_batch: t.List[t.Dict[str, torch.Tensor]],
    device: str,
):
    samples = copy.deepcopy(samples)
    tilts = copy.deepcopy(tilts)
    targets = copy.deepcopy(targets)
    prompts = copy.deepcopy(prompts)
    gt_3d_batch = copy.deepcopy(gt_3d_batch)

    # prepare sample
    samples = samples.to(device)

    # prepare group_start_indices
    group_sizes = [len(t) for t in tilts]
    group_start_indices = [0]
    cumulative_count = 0
    for size in group_sizes[:-1]:
        cumulative_count += size
        group_start_indices.append(cumulative_count)

    # prepare tilts
    tilts = torch.cat(tilts).to(device)

    # prepare gt_3d_batch
    gt_3d_batch = [
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in gt.items()}
        for gt in gt_3d_batch
    ]

    # prepare tomogram_sizes (already in device)
    tomogram_sizes = torch.stack([gt['tomogram_size'] for gt in gt_3d_batch])

    # prepare boxes and categories
    boxes = []
    categories = []
    for tilt_prompts in prompts:
        for prompt in tilt_prompts:
            if torch.is_tensor(prompt['boxes']):  # otherwise, is an empty list
                prompt['boxes'] = prompt['boxes'].to(device)
            if torch.is_tensor(prompt['labels']):  # otherwise, is an empty list
                prompt['labels'] = prompt['labels'].to(device)
            boxes.append(prompt['boxes'])
            categories.append(prompt['labels'])

    return samples, tilts, boxes, categories, tomogram_sizes, group_start_indices, gt_3d_batch


def prepare_input_data_without_prompts(
    samples: torch.Tensor,
    tilts: t.List[torch.Tensor],
    gt_3d_batch: t.List[t.Dict[str, torch.Tensor]],
    device: str,
):
    samples = copy.deepcopy(samples)
    tilts = copy.deepcopy(tilts)
    gt_3d_batch = copy.deepcopy(gt_3d_batch)

    # prepare sample
    samples = samples.to(device)

    # prepare group_start_indices
    group_sizes = [len(t) for t in tilts]
    group_start_indices = [0]
    cumulative_count = 0
    for size in group_sizes[:-1]:
        cumulative_count += size
        group_start_indices.append(cumulative_count)

    # prepare tilts
    tilts = torch.cat(tilts).to(device)

    # prepare gt_3d_batch
    gt_3d_batch = [
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in gt.items()}
        for gt in gt_3d_batch
    ]

    # prepare tomogram_sizes (already in device)
    tomogram_sizes = torch.stack([gt['tomogram_size'] for gt in gt_3d_batch])

    return samples, tilts, tomogram_sizes, group_start_indices, gt_3d_batch
