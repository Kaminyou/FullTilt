# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import sys
import typing as t

import torch

import fulltilt.util.misc as utils
import fulltilt.util.eval_3d as eval_3d


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    processor_3d: torch.nn.Module,
    data_loader: t.Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lr_scheduler=None,
    args=None,
    logger=None,
):
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples, tilts, targets, prompts, gt_3d_batch in metric_logger.log_every(
        data_loader, 10, header, logger=logger,
    ):

        samples = samples.to(device)

        # tilt/target/prompt processing logic
        group_sizes = [len(t) for t in tilts]
        group_start_indices = [0]
        cumulative_count = 0
        for size in group_sizes[:-1]:
            cumulative_count += size
            group_start_indices.append(cumulative_count)

        tilts = torch.cat(tilts).to(device)

        gt_3d_batch = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in gt.items()}
            for gt in gt_3d_batch
        ]

        tomogram_sizes = torch.stack([gt['tomogram_size'] for gt in gt_3d_batch])

        temp_targets = []
        boxes = []
        categories = []
        for _targets, _prompts in zip(targets, prompts):
            _targets = [
                {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in _targets
            ]
            temp_targets += _targets
            for prompt in _prompts:
                if torch.is_tensor(prompt['boxes']):
                    prompt['boxes'] = prompt['boxes'].to(device)
                if torch.is_tensor(prompt['labels']):
                    prompt['labels'] = prompt['labels'].to(device)
                boxes.append(prompt['boxes'])
                categories.append(prompt['labels'])

        targets = temp_targets

        # forward pass
        with torch.amp.autocast('cuda', enabled=args.amp):
            outputs = model(
                samples,
                tilt_angles=tilts,
                group_start_indices=group_start_indices,
                prompts=boxes,
                categories=categories,
                tomogram_sizes=tomogram_sizes,
                gt_3d_batch=gt_3d_batch,  # only for training
            )
            outputs['gt_3d'] = gt_3d_batch
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled,
        )
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        # calculate mAP05r mAP1r
        results_3d = processor_3d(outputs, tomogram_sizes)
        scaled_gt_3d_batch = processor_3d.scale_back_gt(gt_3d_batch)
        metrics_apr = eval_3d.evaluate_3d_batch_apr(
            results_3d,
            scaled_gt_3d_batch,
            score_threshold=0.0,
            r_thresholds=[1.0, 0.5],
            label_mapping=None,
            to_print=False,
        )
        ap_1r = []
        ap_05r = []
        for _, v in metrics_apr['per_class'].items():
            ap_1r.append(v[1.0])
            ap_05r.append(v[0.5])
        metric_logger.update(mAP_1r=torch.tensor(ap_1r).mean().item())
        metric_logger.update(mAP_05r=torch.tensor(ap_05r).mean().item())

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})

    return resstat


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    processor_3d: torch.nn.Module,
    data_loader: t.Iterable,
    device: str,
    args=None,
    logger=None,
):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    for samples, tilts, targets, prompts, gt_3d_batch in metric_logger.log_every(
        data_loader, 10, header, logger=logger,
    ):
        samples = samples.to(device)

        group_sizes = [len(t) for t in tilts]
        group_start_indices = [0]
        cumulative_count = 0
        for size in group_sizes[:-1]:
            cumulative_count += size
            group_start_indices.append(cumulative_count)

        tilts = torch.cat(tilts).to(device)

        gt_3d_batch = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in gt.items()}
            for gt in gt_3d_batch
        ]

        tomogram_sizes = torch.stack([gt['tomogram_size'] for gt in gt_3d_batch])

        temp_targets = []
        boxes = []
        categories = []
        for _targets, _prompts in zip(targets, prompts):
            _targets = [
                {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in _targets
            ]
            temp_targets += _targets
            for prompt in _prompts:
                if torch.is_tensor(prompt['boxes']):
                    prompt['boxes'] = prompt['boxes'].to(device)
                if torch.is_tensor(prompt['labels']):
                    prompt['labels'] = prompt['labels'].to(device)
                boxes.append(prompt['boxes'])
                categories.append(prompt['labels'])

        targets = temp_targets

        with torch.amp.autocast('cuda', enabled=args.amp):

            outputs = model(
                samples,
                tilt_angles=tilts,
                group_start_indices=group_start_indices,
                prompts=boxes,
                categories=categories,
                tomogram_sizes=tomogram_sizes,
                force_aux_interm_loss=True,  # Calculate aux/interm loss during evaluation
            )
            outputs['gt_3d'] = gt_3d_batch
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

        # Reduce and log loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        # calculate mAP05r mAP1r
        results_3d = processor_3d(outputs, tomogram_sizes)
        scaled_gt_3d_batch = processor_3d.scale_back_gt(gt_3d_batch)
        metrics_apr = eval_3d.evaluate_3d_batch_apr(
            results_3d,
            scaled_gt_3d_batch,
            score_threshold=0.0,
            r_thresholds=[1.0, 0.5],
            label_mapping=None,
            to_print=False,
        )
        ap_1r = []
        ap_05r = []
        for _, v in metrics_apr['per_class'].items():
            ap_1r.append(v[1.0])
            ap_05r.append(v[0.5])
        metric_logger.update(mAP_1r=torch.tensor(ap_1r).mean().item())
        metric_logger.update(mAP_05r=torch.tensor(ap_05r).mean().item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats
