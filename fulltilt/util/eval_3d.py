# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.spatial import KDTree


def scale_back_gt(
    gt_batch: t.List[t.Dict[str, torch.Tensor]],
) -> t.List[t.Dict[str, torch.Tensor]]:
    """
    Normalized GT [0, 1] to physical GT
    Args:
        gt_batch: List[Dict]
    Returns:
        List[Dict]
    """
    scaled_gt = []
    for gt in gt_batch:
        boxes = gt['boxes_3d'].clone()
        labels = gt['labels'].clone()

        size = gt['tomogram_size']
        if isinstance(size, torch.Tensor):
            size = size.to(boxes.device)
        W, H, D = size[0], size[1], size[2]

        scale = torch.tensor([W, H, D, W, H], device=boxes.device)

        boxes_abs = boxes * scale
        scaled_gt.append({
            'boxes_3d': boxes_abs,
            'labels': labels,
            'tomogram_size': size,
        })
    return scaled_gt


def boxes_to_corners_3d(
    boxes_5d: torch.Tensor,
    ceil_radius: bool = False,
) -> torch.Tensor:
    """
    [cx, cy, cz, w, h] to [x1, y1, z1, x2, y2, z2]
    diameter = (w + h) / 2 for each side length
    """
    cx, cy, cz, w, h = boxes_5d.unbind(-1)

    diameter = (w + h) / 2.0
    radius = diameter / 2.0
    if ceil_radius:
        radius = torch.ceil(radius)

    x1 = cx - radius
    y1 = cy - radius
    z1 = cz - radius

    x2 = cx + radius
    y2 = cy + radius
    z2 = cz + radius

    return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)


def box3d_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    ceil_radius: bool = False,
) -> torch.Tensor:
    """
    3D IoU for 3D bbox
    Args:
        boxes1: [N, 5] (cx, cy, cz, w, h) - Predictions
        boxes2: [M, 5] (cx, cy, cz, w, h) - Ground Truths
    Returns:
        iou: [N, M]
    """
    b1 = boxes_to_corners_3d(boxes1, ceil_radius=ceil_radius)
    b2 = boxes_to_corners_3d(boxes2, ceil_radius=ceil_radius)

    # Intersection
    # intersection_min = max(min1, min2)
    inter_min = torch.max(b1[:, None, :3], b2[None, :, :3])  # [N, M, 3]
    # intersection_max = min(max1, max2)
    inter_max = torch.min(b1[:, None, 3:], b2[None, :, 3:])  # [N, M, 3]

    # volume = multiply (max - min).clamp(0)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]  # [N, M]

    # Union
    # vol = (x2-x1) * (y2-y1) * (z2-z1)
    vol1 = (b1[:, 3] - b1[:, 0]) * (b1[:, 4] - b1[:, 1]) * (b1[:, 5] - b1[:, 2])  # [N]
    vol2 = (b2[:, 3] - b2[:, 0]) * (b2[:, 4] - b2[:, 1]) * (b2[:, 5] - b2[:, 2])  # [M]

    union_vol = vol1[:, None] + vol2[None, :] - inter_vol

    # 4. IoU
    iou = inter_vol / (union_vol + 1e-8)
    return iou


def compute_ap_per_class(
    tp: torch.Tensor,
    conf: torch.Tensor,
    num_gt: int,
):
    i = torch.argsort(conf, descending=True)
    tp = tp[i]
    fp = 1 - tp
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / (num_gt + 1e-8)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = torch.cat(
        (
            torch.tensor([0.0], device=recalls.device),
            recalls,
            torch.tensor([1.0], device=recalls.device)
        )
    )
    precisions = torch.cat(
        (
            torch.tensor([0.0], device=precisions.device),
            precisions,
            torch.tensor([0.0], device=precisions.device)
        )
    )
    for i in range(precisions.shape[0] - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


@torch.no_grad()
def evaluate_3d_batch(
    results_batch: t.List[t.Dict[str, torch.Tensor]],
    gt_batch: t.List[t.Dict[str, torch.Tensor]],
    score_threshold: float = 0.05,
    iou_thresholds: t.Optional[t.List[float]] = None,
    is_gt_batch_normalized: bool = False,  # if true, need to scale back
    label_mapping: t.Optional[t.Dict[int, str]] = None,  # map int label to string
    to_print: bool = True,
    ceil_radius: bool = False,
):
    """
    Calculate mAP
    Args:
        results_batch: prediction
        gt_batch: ground Truth
        score_threshold: filter predictions that have a score lower than score_threshold
        iou_thresholds: (List[float] or None) if None, use standard [0.50, 0.55, ..., 0.95]
    """
    if iou_thresholds is None:
        # COCO Standard
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    else:
        # User Defined
        iou_thresholds = [round(float(x), 2) for x in iou_thresholds]

    # scale GT back to physical coordinates
    if is_gt_batch_normalized:
        gt_batch_abs = scale_back_gt(gt_batch)
    else:
        gt_batch_abs = gt_batch

    # collect all labels
    all_labels = set()
    for gt in gt_batch_abs:
        all_labels.update(gt['labels'].cpu().tolist())
    all_labels = sorted(list(all_labels))

    # count # of gt per class
    num_gt_per_class = {label: 0 for label in all_labels}
    for gt in gt_batch_abs:
        for label in all_labels:
            num_gt_per_class[label] += (gt['labels'] == label).sum().item()

    # match_stats[label][iou_threshold] = {'tp': [], 'conf': []}
    match_stats = {
        label: {t: {'tp': [], 'conf': []} for t in iou_thresholds} for label in all_labels
    }

    # iter batch
    for pred, gt in zip(results_batch, gt_batch_abs):

        # filtered by score threshold
        valid_mask = pred['scores'] >= score_threshold
        pred_boxes_all = pred['boxes_3d'][valid_mask]
        pred_scores_all = pred['scores'][valid_mask]
        pred_labels_all = pred['labels'][valid_mask]

        gt_boxes_all = gt['boxes_3d']
        gt_labels_all = gt['labels']

        for label in all_labels:
            p_mask = pred_labels_all == label
            g_mask = gt_labels_all == label

            p_box = pred_boxes_all[p_mask]
            p_score = pred_scores_all[p_mask]
            g_box = gt_boxes_all[g_mask]

            if len(p_box) == 0:
                continue

            if len(g_box) == 0:
                for threshold in iou_thresholds:
                    match_stats[label][threshold]['tp'].append(
                        torch.zeros(len(p_score), device=p_score.device)
                    )
                    match_stats[label][threshold]['conf'].append(p_score)
                continue

            iou_matrix = box3d_iou(p_box, g_box, ceil_radius=ceil_radius)

            sort_idx = torch.argsort(p_score, descending=True)
            iou_matrix_sorted = iou_matrix[sort_idx]

            for threshold in iou_thresholds:
                tp = torch.zeros(len(p_score), device=p_score.device)
                gt_matched = torch.zeros(len(g_box), dtype=torch.bool, device=g_box.device)

                for i in range(len(p_score)):
                    max_iou, max_idx = torch.max(iou_matrix_sorted[i], dim=0)

                    if max_iou >= threshold:
                        if not gt_matched[max_idx]:
                            tp[sort_idx[i]] = 1.0
                            gt_matched[max_idx] = True

                match_stats[label][threshold]['tp'].append(tp)
                match_stats[label][threshold]['conf'].append(p_score)

    # ap_data[label][iou_threshold] = AP_value
    ap_data = {label: {threshold: 0.0 for threshold in iou_thresholds} for label in all_labels}

    # Header
    # TODO: deal with COCO Standard
    header_thresholds = [t for t in iou_thresholds]  # if t in [0.25, 0.5, 0.75]
    if not header_thresholds:
        header_thresholds = [
            iou_thresholds[0], iou_thresholds[-1]
        ] if len(iou_thresholds) > 1 else iou_thresholds

    col_width = 10
    if label_mapping is not None:
        max_label_len = max(len(str(label_mapping[label])) for label in label_mapping)
        col_width = max(max_label_len + 2, 20)

    if to_print:
        print("-" * 65)
        header_str = " | ".join([f"AP_{int(t*100):<2}" for t in header_thresholds])
        print(f"{'Class':<{col_width}} | {'mAP':<8} | {header_str} | GT_Count")
        print("-" * 65)

    all_maps = []  # Mean over all requested thresholds
    for label in all_labels:
        if num_gt_per_class[label] == 0:
            continue

        class_aps = []
        for threshold in iou_thresholds:
            tps_list = match_stats[label][threshold]['tp']
            confs_list = match_stats[label][threshold]['conf']

            if len(tps_list) == 0:
                ap = torch.tensor(0.0, device=results_batch[0]['scores'].device)
            else:
                tp_all = torch.cat(tps_list)
                conf_all = torch.cat(confs_list)
                ap = compute_ap_per_class(tp_all, conf_all, num_gt_per_class[label])

            ap_data[label][threshold] = ap.item()
            class_aps.append(ap.item())

        m_ap_class = np.mean(class_aps)
        all_maps.append(m_ap_class)

        ap_strs = []
        for threshold in header_thresholds:
            val = ap_data[label].get(threshold, 0.0)
            ap_strs.append(f"{val:.4f}")
        ap_display = " | ".join(ap_strs)
        if to_print:
            if label_mapping is None:
                print(
                    f"{label:<{col_width}} | {m_ap_class:.4f}   | {ap_display}"
                    f" | {num_gt_per_class[label]}"
                )
            else:
                print(
                    f"{label_mapping[label]:<{col_width}} | {m_ap_class:.4f}   |"
                    f" {ap_display} | {num_gt_per_class[label]}"
                )

    mAP_final = np.mean(all_maps) if all_maps else 0.0

    if to_print:
        print("-" * 65)
        print(f"{'MEAN':<{col_width}} | {mAP_final:.4f}")

    mAP_50 = np.mean([ap_data[label].get(0.5, np.nan) for label in all_labels])
    mAP_75 = np.mean([ap_data[label].get(0.75, np.nan) for label in all_labels])

    return {
        'mAP': mAP_final,  # Mean of ALL provided thresholds
        'mAP_50': mAP_50 if not np.isnan(mAP_50) else None,
        'mAP_75': mAP_75 if not np.isnan(mAP_75) else None,
        'per_class': ap_data,
        'thresholds_used': iou_thresholds
    }


def compute_distance_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        boxes1: [N, 5] (cx, cy, cz, w, h) - Predictions
        boxes2: [M, 5] (cx, cy, cz, w, h) - Ground Truths
    Returns:
        dist: [N, M]
    """
    p_centers = boxes1[:, :3]
    g_centers = boxes2[:, :3]

    dist = torch.cdist(p_centers, g_centers, p=2.0)
    return dist


@torch.no_grad()
def evaluate_3d_batch_apr(
    results_batch: t.List[t.Dict[str, torch.Tensor]],
    gt_batch: t.List[t.Dict[str, torch.Tensor]],
    score_threshold: float = 0.05,
    r_thresholds: t.Optional[t.List[float]] = None,
    is_gt_batch_normalized: bool = False,
    label_mapping: t.Optional[t.Dict[int, str]] = None,
    to_print: bool = True,
    sort_by_size: bool = True,
    ceil_radius: bool = False,
):
    if r_thresholds is None:
        r_thresholds = [1.0, 0.5]
    else:
        r_thresholds = [round(float(x), 2) for x in r_thresholds]

    if is_gt_batch_normalized:
        gt_batch_abs = scale_back_gt(gt_batch)
    else:
        gt_batch_abs = gt_batch

    all_labels = sorted(
        list(set(label for gt in gt_batch_abs for label in gt['labels'].cpu().tolist()))
    )
    num_gt_per_class = {
        label: sum(
            (gt['labels'] == label).sum().item() for gt in gt_batch_abs
        ) for label in all_labels
    }

    class_radius = {label: 0.0 for label in all_labels}

    match_stats = {
        label: {rt: {'tp': [], 'conf': []} for rt in r_thresholds} for label in all_labels
    }

    for pred, gt in zip(results_batch, gt_batch_abs):
        valid_mask = pred['scores'] >= score_threshold
        pred_boxes_all = pred['boxes_3d'][valid_mask]
        pred_scores_all = pred['scores'][valid_mask]
        pred_labels_all = pred['labels'][valid_mask]

        gt_boxes_all = gt['boxes_3d']
        gt_labels_all = gt['labels']

        for label in all_labels:
            p_mask = pred_labels_all == label
            g_mask = gt_labels_all == label

            p_box = pred_boxes_all[p_mask]
            p_score = pred_scores_all[p_mask]
            g_box = gt_boxes_all[g_mask]

            if len(g_box) > 0:
                current_radius = torch.mean((g_box[:, 3] + g_box[:, 4]) / 4.0)
                if ceil_radius:
                    current_radius = torch.ceil(current_radius)
                current_radius = current_radius.item()

                class_radius[label] = current_radius

            if len(p_box) == 0:
                continue

            if len(g_box) == 0:
                for rt in r_thresholds:
                    match_stats[label][rt]['tp'].append(
                        torch.zeros(len(p_score), device=p_score.device)
                    )
                    match_stats[label][rt]['conf'].append(p_score)
                continue

            dist_matrix = compute_distance_3d(p_box, g_box)
            gt_radii = (g_box[:, 3] + g_box[:, 4]) / 4.0

            sort_idx = torch.argsort(p_score, descending=True)
            dist_matrix_sorted = dist_matrix[sort_idx]
            p_score_sorted = p_score[sort_idx]

            for rt in r_thresholds:
                tp = torch.zeros(len(p_score), device=p_score.device)
                gt_matched = torch.zeros(len(g_box), dtype=torch.bool, device=g_box.device)

                dist_limits = gt_radii * rt

                for i in range(len(p_score_sorted)):
                    dists_to_gts = dist_matrix_sorted[i]
                    is_within_radius = dists_to_gts <= dist_limits

                    if torch.any(is_within_radius):
                        tmp_dist = dists_to_gts.clone()
                        tmp_dist[~is_within_radius] = float('inf')
                        tmp_dist[gt_matched] = float('inf')

                        min_dist, max_idx = torch.min(tmp_dist, dim=0)

                        if min_dist != float('inf'):
                            tp[i] = 1.0
                            gt_matched[max_idx] = True

                match_stats[label][rt]['tp'].append(tp)
                match_stats[label][rt]['conf'].append(p_score_sorted)

    display_labels = all_labels
    if sort_by_size:
        display_labels = sorted(all_labels, key=lambda label: class_radius[label])

    ap_data = {label: {rt: 0.0 for rt in r_thresholds} for label in all_labels}
    all_maps = []

    if to_print:
        label_col_width = max(
            len(str(label_mapping[label])) if label_mapping else 0 for label in all_labels
        )
        label_col_width = max(label_col_width + 2, 20)
        print("-" * 95)
        header_str = " | ".join([f"AP@{rt}r" for rt in r_thresholds])
        print(
            f"{'Class':<{label_col_width}} | {'Radius':<8} | {'mAP':<8} | {header_str} | GT_Count"
        )
        print("-" * 95)

    for label in display_labels:
        if num_gt_per_class[label] == 0:
            continue
        class_aps = []
        for rt in r_thresholds:
            if len(match_stats[label][rt]['tp']) == 0:
                ap = 0.0
            else:
                tp_all = torch.cat(match_stats[label][rt]['tp'])
                conf_all = torch.cat(match_stats[label][rt]['conf'])
                ap = compute_ap_per_class(tp_all, conf_all, num_gt_per_class[label]).item()

            ap_data[label][rt] = ap
            class_aps.append(ap)

        m_ap_class = np.mean(class_aps)
        all_maps.append(m_ap_class)

        if to_print:
            ap_display = " | ".join([f"{ap_data[label][rt]:.4f}" for rt in r_thresholds])
            name = label_mapping[label] if label_mapping and label in label_mapping else str(label)
            print(
                f"{name:<{label_col_width}} | {class_radius[label]:<8.2f} | {m_ap_class:.4f}   |"
                f" {ap_display} | {num_gt_per_class[label]}"
            )

    mAP_final = np.mean(all_maps) if all_maps else 0.0
    if to_print:
        print("-" * 95)
        print(f"{'MEAN':<{label_col_width}} | {'':<8} | {mAP_final:.4f}")

    return {
        'mAP': mAP_final,
        'per_class': ap_data,
        'thresholds_used': r_thresholds,
        'class_radius': class_radius
    }


def compute_metrics(
    reference_points: npt.NDArray[float],
    reference_radius: float,
    candidate_points: npt.NDArray[float],
    allow_duplicate: bool = True,
) -> t.Tuple[int, int, int]:

    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    # TODO: this is incorrect: if one particle is close to several gt => count all gt => tp increase
    for match in raw_matches:
        if len(match) >= 2 and not allow_duplicate:
            raise ValueError('Over then 1 reference match a single candidate.')
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def compute_metrics_correct(
    reference_points: npt.NDArray[float],  # GT points
    reference_radius: float,
    candidate_points: npt.NDArray[float],  # Pred points
    candidate_scores: npt.NDArray[float],  # Pred scores
    optimial_match: bool = True,
) -> t.Tuple[int, int, int]:
    num_gt = len(reference_points)
    num_pred = len(candidate_points)

    if num_gt == 0:
        return 0, num_pred, 0
    if num_pred == 0:
        return 0, 0, num_gt

    # 1. Sort predictions by score descending
    sort_idx = np.argsort(-candidate_scores)
    candidate_points = candidate_points.copy()
    candidate_points = candidate_points[sort_idx]

    # 2. Build KDTree for Ground Truths
    ref_tree = KDTree(reference_points)

    # 3. Track which GTs have been "used"
    gt_matched = np.zeros(num_gt, dtype=bool)
    tp = 0

    # 4. Greedy matching based on score priority
    # For each prediction (highest score first), find all GTs within radius
    raw_matches = ref_tree.query_ball_point(candidate_points, r=reference_radius)

    for i, matches in enumerate(raw_matches):
        if len(matches) > 0:
            # If there are multiple GTs in radius, pick the closest one
            # that hasn't been matched yet.
            unmatched_gts = [m for m in matches if not gt_matched[m]]

            if unmatched_gts:
                # Optimal version: Find the absolute closest among unmatched
                if optimial_match:
                    distances = np.linalg.norm(
                        reference_points[unmatched_gts] - candidate_points[i], axis=1,
                    )
                    best_gt_idx = unmatched_gts[np.argmin(distances)]

                # Simple version: Pick the first available GT in the match list
                else:
                    best_gt_idx = unmatched_gts[0]

                tp += 1
                gt_matched[best_gt_idx] = True

    fp = num_pred - tp
    fn = num_gt - tp

    return tp, fp, fn


@torch.no_grad()
def evaluate_3d_batch_confusion(
    results_batch: t.List[t.Dict[str, torch.Tensor]],
    gt_batch: t.List[t.Dict[str, torch.Tensor]],
    score_threshold: float = 0.05,
    size: t.Optional[t.Dict[int, float]] = None,  # radius
    is_gt_batch_normalized: bool = False,
    label_mapping: t.Optional[t.Dict[int, str]] = None,
    to_print: bool = True,
    beta: float = 1.0,
    sort_by_size: bool = True,
    allow_duplicate: bool = True,
    ceil_radius: bool = False,
    use_czii_compute_metrics: bool = False,
    optimial_match: bool = True,  # only works when use_czii_compute_metrics is False
) -> t.Dict:
    if is_gt_batch_normalized:
        gt_batch_abs = scale_back_gt(gt_batch)
    else:
        gt_batch_abs = gt_batch

    all_labels = set()
    for gt in gt_batch_abs:
        all_labels.update(gt['labels'].cpu().tolist())
    all_labels = sorted(list(all_labels))

    stats_per_class = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in all_labels}
    class_radius = {label: 0.0 for label in all_labels}

    for pred, gt in zip(results_batch, gt_batch_abs):
        valid_mask = pred['scores'] >= score_threshold
        p_boxes_all = pred['boxes_3d'][valid_mask].cpu().numpy()
        p_labels_all = pred['labels'][valid_mask].cpu().numpy()
        p_scores_all = pred['scores'][valid_mask].cpu().numpy()

        g_boxes_all = gt['boxes_3d'].cpu().numpy()
        g_labels_all = gt['labels'].cpu().numpy()

        for label in all_labels:
            p_points = p_boxes_all[p_labels_all == label][:, :3]
            p_scores = p_scores_all[p_labels_all == label]
            g_boxes_class = g_boxes_all[g_labels_all == label]
            g_points = g_boxes_class[:, :3]

            if size is not None and label in size:
                radius = size[label]
            else:
                if len(g_boxes_class) > 0:
                    radius = np.mean((g_boxes_class[:, 3] + g_boxes_class[:, 4]) / 4.0)
                    if ceil_radius:
                        radius = np.ceil(radius)
                else:
                    radius = 0.0

            if radius > 0:
                class_radius[label] = radius

            if use_czii_compute_metrics:
                tp, fp, fn = compute_metrics(
                    g_points, radius, p_points, allow_duplicate=allow_duplicate,
                )
            else:
                tp, fp, fn = compute_metrics_correct(
                    g_points,
                    radius,
                    p_points,
                    candidate_scores=p_scores,
                    optimial_match=optimial_match,
                )

            stats_per_class[label]['tp'] += tp
            stats_per_class[label]['fp'] += fp
            stats_per_class[label]['fn'] += fn

    display_labels = all_labels
    if sort_by_size:
        display_labels = sorted(all_labels, key=lambda label: class_radius[label])

    results = {'per_class': {}}
    col_width = 20

    if to_print:
        print("-" * 110)
        print(
            f"{'Class':<{col_width}} | {'Radius':<10} | {'Precision':<10} | {'Recall':<10} | "
            f"{'F-beta':<10} | TP / FP / FN"
        )
        print("-" * 110)

    for label in display_labels:
        tp = stats_per_class[label]['tp']
        fp = stats_per_class[label]['fp']
        fn = stats_per_class[label]['fn']
        radius_val = class_radius[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fbeta = (
            (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        ) if (precision + recall) > 0 else 0.0

        results['per_class'][label] = {
            'precision': precision, 'recall': recall, 'fbeta': fbeta,
            'tp': tp, 'fp': fp, 'fn': fn, 'radius': radius_val
        }

        if to_print:
            name = label_mapping[label] if label_mapping and label in label_mapping else str(label)
            print(
                f"{name:<{col_width}} | {radius_val:<10.2f} | {precision:.4f}    |"
                f" {recall:.4f}    | {fbeta:.4f}    | {tp} / {fp} / {fn}"
            )

    return results


def summarize_metrics(
    metric_collection: t.Dict[str, t.Dict[str, t.List[float]]],  # {metric: {class: [values]}}
) -> pd.DataFrame:
    records = []
    for metric, labels in metric_collection.items():
        for label, values in labels.items():
            arr = np.array(values)
            records.append({
                'Metric': metric,
                'Label': label,
                'mean': np.mean(arr),
                'std': np.std(arr),
                # 'min': np.min(arr),
                # 'max': np.max(arr)
            })

    df = pd.DataFrame(records)

    df_long = df.melt(id_vars=['Metric', 'Label'], var_name='Stat', value_name='Val')

    df_pivot = df_long.pivot_table(index='Label', columns=['Metric', 'Stat'], values='Val')

    stat_order = ['mean', 'std']
    df_pivot = df_pivot.reindex(columns=stat_order, level='Stat')

    return df_pivot


def summarize_metrics_formatted(
    metric_collection: t.Dict[str, t.Dict[str, t.List[float]]],  # {metric: {class: [values]}}
    precision: int = 3,
) -> pd.DataFrame:
    records = []
    for metric, labels in metric_collection.items():
        for label, values in labels.items():
            arr = np.array(values)

            mean_val = np.mean(arr)
            std_val = np.std(arr)

            formatted_val = f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"

            records.append({
                'Metric': metric,
                'Label': label,
                'Value': formatted_val
            })

    df = pd.DataFrame(records)
    df_pivot = df.pivot(index='Label', columns='Metric', values='Value')
    return df_pivot
