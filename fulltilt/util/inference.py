# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import typing as t

import cv2
import numpy as np
import numpy.typing as npt
import torch

from fulltilt.datasets import transforms
from fulltilt.models import registry
from fulltilt.models.FullTilt import fulltilt
from fulltilt.util import misc, slconfig


def xyz_to_xy_bbox(
    xyz_indices: npt.NDArray[float],
    diameters: npt.NDArray[float],
) -> npt.NDArray[float]:
    """
    Converts 3D center coordinates to 2D bounding boxes for projection onto the XY plane.

    Args:
        xyz_indices (npt.NDArray[float]): A 2D array of shape (N, 3) containing 3D coordinates
            [x, y, z].
        diameters (npt.NDArray[float]): A 1D array of shape (N,) containing the diameters
            corresponding to each point.

    Returns:
        npt.NDArray[float]: A 2D array of shape (N, 4) representing 2D bounding boxes in
            [x1, y1, x2, y2] format.
    """
    x1 = xyz_indices[:, 0] - diameters / 2
    x2 = xyz_indices[:, 0] + diameters / 2
    y1 = xyz_indices[:, 1] - diameters / 2
    y2 = xyz_indices[:, 1] + diameters / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def rotate_coordinate(
    xyz_indices: npt.NDArray[t.Union[int, float]],
    theta: float,
    W: int,
    D: int,
) -> npt.NDArray[float]:
    """
    Rotates 3D coordinates around the Y-axis (tilt axis) by a given angle.

    The rotation origin is assumed to be the center of the XZ plane, which corresponds
    to the center of the tomogram slice (W/2, D/2).

    Args:
        xyz_indices (npt.NDArray[t.Union[int, float]]): A 2D array of shape (N, 3) containing
            [x, y, z] coordinates.
        theta (float): The tilt angle in degrees.
        W (int): The width of the tomogram.
        D (int): The depth (thickness) of the tomogram.

    Returns:
        npt.NDArray[float]: A 2D array of shape (N, 3) containing the rotated coordinates, rounded
            to the nearest integer.
    """
    xyz_indices = xyz_indices.copy().astype(float)
    x = xyz_indices[:, 0]
    z = xyz_indices[:, 2]

    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_rot = (x - W / 2) * cos_t + (z - D / 2) * sin_t + W / 2
    z_rot = -(x - W / 2) * sin_t + (z - D / 2) * cos_t + D / 2

    rot_xyz_indices = np.zeros_like(xyz_indices)
    rot_xyz_indices[:, 0] = x_rot
    rot_xyz_indices[:, 1] = xyz_indices[:, 1]
    rot_xyz_indices[:, 2] = z_rot

    return rot_xyz_indices


def init_model(
    config_file: str,
    checkpoint_path: str,
    device: str,
    num_queries: int = 900,
):
    """
    Initializes the FullTilt PyTorch model, loads checkpoint weights, and builds the 3D
    post-processor.

    Args:
        config_file (str): Path to the configuration file (e.g., 'configs/cfg_cryo.py').
        checkpoint_path (str): Path to the saved model checkpoint file (.pth).
        device (str): Computation device to load the model onto (e.g., 'cuda' or 'cpu').
        num_queries (int): The number of queries to use for the model. Default is 900.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The initialized FullTilt network in evaluation mode.
            - processor_3d (torch.nn.Module): The post-processing module for 3D bounding boxes.
            - args (Any): The parsed arguments/configurations object.
    """
    class Args:
        def __init__(self):
            self.config_file = config_file
            self.options = None
            self.device = device
            self.amp = True
    args = Args()

    cfg = slconfig.SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError(f'Key {k} can used by args only')

    args.two_stage_type = 'standard'
    args.num_queries = num_queries

    build_func = registry.MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # To support loading checkpoints with different number of queries than the model's
    # expected number.
    misc.fix_num_query_weight(model, checkpoint)

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(args.device)
    model.eval()
    processor_3d = fulltilt.PostProcess3D(
        # NOTE: Assume only one class and we want to keep all queries for post-processing.
        num_select=args.num_queries,
        one_class_per_proposal=False,
    )

    return model, processor_3d, args


def verify_inputs(
    tomogram_whd: t.Tuple[int, int, int],
    tiltseries: npt.NDArray[float],
    thetas: t.List[float],
    prompts_xyz: t.Optional[npt.NDArray[float]],
    prompts_label: t.Optional[t.List[int]],
    diameters: t.Optional[npt.NDArray[float]],
    device: str,
    confidence_score_threshold: t.Union[float, t.Dict[int, float]],
    is_inference_with_prototypes: bool = False,
):
    """
    Verifies that the input data dimensions and types are consistent and valid for the inference
    pipeline.

    Args:
        tomogram_whd (Tuple[int, int, int]): The dimensions of the target tomogram (Width, Height,
            Depth).
        tiltseries (npt.NDArray[float]): The input tilt-series images of shape (C, H, W).
        thetas (List[float]): A list of tilt angles corresponding to each image channel in the
            tilt-series.
        prompts_xyz (t.Optional[npt.NDArray[float]]): Initial 3D point prompts of shape (N, 3) in
            [x, y, z]
        prompts_label (t.Optional[t.List[int]]): Class labels corresponding to the point prompts.
        diameters (t.Optional[npt.NDArray[float]]): Initial estimated diameters for the prompts.
        device (str): The device for model computation ('cuda' or 'cpu').
        confidence_score_threshold (Union[float, Dict[int, float]]): The confidence score threshold
            for filtering predictions. Can be a single float applied to all classes or a dictionary
            mapping class IDs to thresholds.
        is_inference_with_prototypes (bool): Whether the verification is for the pipeline that
            takes prototypes as input.
    Raises:
        ValueError: If any of the input dimensions or types are inconsistent or invalid.
    """
    if not isinstance(tomogram_whd, tuple) or len(tomogram_whd) != 3:
        raise ValueError(f'tomogram_whd must be a tuple of (W, H, D), got {tomogram_whd}')

    if tiltseries.ndim != 3:
        raise ValueError(
            f'tiltseries must be a 3D array of shape (C, H, W), got {tiltseries.shape}'
        )

    if not isinstance(thetas, list):
        raise ValueError(f'thetas must be a list of floats, got {type(thetas)}')

    if len(thetas) != tiltseries.shape[0]:
        raise ValueError(
            'Number of tilt angles must match number of channels in tiltseries,'
            f' got {len(thetas)} and {tiltseries.shape[0]}'
        )

    if not is_inference_with_prototypes:
        if prompts_xyz.ndim != 2 or prompts_xyz.shape[1] != 3:
            raise ValueError(
                f'prompts_xyz must be a 2D array of shape (N, 3), got {prompts_xyz.shape}'
            )

    if not is_inference_with_prototypes:
        if len(prompts_label) != prompts_xyz.shape[0]:
            raise ValueError(
                'Number of prompt labels must match number of prompt coordinates,'
                f' got {len(prompts_label)} and {prompts_xyz.shape[0]}')
    if not is_inference_with_prototypes:
        if diameters.ndim != 1 or diameters.shape[0] != prompts_xyz.shape[0]:
            raise ValueError(
                'diameters must be a 1D array with the same length as the number of prompts,'
                f' got {diameters.shape} and {prompts_xyz.shape[0]}'
            )

    if device not in ['cuda', 'cpu']:
        raise ValueError(f'device must be either "cuda" or "cpu", got {device}')

    if not (
        isinstance(confidence_score_threshold, float)
        or isinstance(confidence_score_threshold, dict)
    ):
        raise ValueError(
            'confidence_score_threshold must be either a float or a dict,'
            f' got {type(confidence_score_threshold)}'
        )

    if isinstance(confidence_score_threshold, dict):
        for k, v in confidence_score_threshold.items():
            if not isinstance(k, int):
                raise ValueError(
                    'Keys of confidence_score_threshold dict must be integers representing class'
                    f' IDs, got key {k} of type {type(k)}'
                )
            if not isinstance(v, float):
                raise ValueError(
                    'Values of confidence_score_threshold dict must be floats representing'
                    f' thresholds, got value {v} of type {type(v)}'
                )

        unique_labels = set(prompts_label)
        unique_labels_in_thresholds = set(confidence_score_threshold.keys())
        if not unique_labels == unique_labels_in_thresholds:
            raise ValueError(
                'When confidence_score_threshold is a dict, its keys must match the unique class'
                f' labels in prompts_label. Got unique labels {unique_labels} and threshold keys'
                f' {unique_labels_in_thresholds}'
            )


def pipeline(
    model: torch.nn.Module,
    processor_3d: torch.nn.Module,
    tomogram_whd: t.Tuple[int, int, int],
    tiltseries: npt.NDArray[float],
    thetas: t.List[float],
    prompts_xyz: npt.NDArray[float],
    prompts_label: t.List[int],
    diameters: npt.NDArray[float],
    device: str,
    num_select: t.Optional[int] = None,
    nms_threshold: float = -1.0,
    confidence_score_threshold: t.Union[float, t.Dict[int, float]] = 0.0,
) -> t.Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], npt.NDArray[float]]:
    """
    Executes the main inference pipeline: takes 3D coordinate prompts, projects them across a
    tilt-series, and predicts final 3D centers, diameters, labels, and confidence scores.

    Args:
        model (torch.nn.Module): The initialized 2D-to-3D detection model.
        processor_3d (torch.nn.Module): The 3D post-processing module.
        tomogram_whd (Tuple[int, int, int]): The dimensions of the target tomogram (Width, Height,
            Depth).
        tiltseries (npt.NDArray[float]): The input tilt-series images of shape (C, H, W).
        thetas (List[float]): A list of tilt angles corresponding to each image channel in the
            tilt-series.
        prompts_xyz (npt.NDArray[float]): Initial 3D point prompts of shape (N, 3) in [x, y, z]
            format.
        prompts_label (List[int]): Class labels corresponding to the point prompts.
        diameters (npt.NDArray[float]): Initial estimated diameters for the prompts.
        device (str): The device for model computation ('cuda' or 'cpu').
        num_select (Optional[int]): The number of top estimates to select for post-processing.
        If None, defaults to model's num_select.
        nms_threshold (float): The threshold for non-maximum suppression. If lower than 0,
            no NMS will be applied. Higher values result in more aggressive suppression. See the
            PostProcess3D class for details.
        confidence_score_threshold (Union[float, Dict[int, float]]): The confidence score threshold
            for filtering predictions. Can be a single float applied to all classes or a dictionary
            mapping class IDs to thresholds.

    Returns:
        tuple: A tuple containing the inference results:
            - estimated_centers (npt.NDArray[float]): Estimated 3D coordinates [x, y, z] of the
                detected objects.
            - estimated_diameters (npt.NDArray[float]): Estimated average 3D diameters.
            - estimated_labels (npt.NDArray[int]): Estimated class labels.
            - confidence_scores (npt.NDArray[float]): Confidence scores for the estimations.
    """
    verify_inputs(
        tomogram_whd=tomogram_whd,
        tiltseries=tiltseries,
        thetas=thetas,
        prompts_xyz=prompts_xyz,
        prompts_label=prompts_label,
        diameters=diameters,
        device=device,
        confidence_score_threshold=confidence_score_threshold,
    )
    w, h, d = tomogram_whd
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    channels = tiltseries.shape[0]
    all_images = []
    all_bboxes = []
    all_categories = []
    all_tilts = []
    for theta, channel_idx in zip(thetas, list(range(channels))):
        image = tiltseries[channel_idx]

        # NOTE: Assuming input tilt-series is in a range that can be safely converted to uint8
        # without clipping.
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        rotated_xyz_indices = rotate_coordinate(xyz_indices=prompts_xyz, theta=theta, W=w, D=d)
        rotated_bboxes = xyz_to_xy_bbox(rotated_xyz_indices, diameters)

        image, bboxes = normalize(image, {'boxes': torch.tensor(rotated_bboxes)})
        all_images.append(image)
        all_bboxes.append(bboxes['boxes'])
        all_categories.append(prompts_label)
        all_tilts.append(theta)

    all_images = misc.nested_tensor_from_tensor_list(all_images)

    all_images = all_images.to(device)
    all_tilts = torch.tensor(all_tilts).to(device)
    all_bboxes = [bboxes.to(device) for bboxes in all_bboxes]
    all_categories = [torch.tensor(categories).to(device) for categories in all_categories]
    tomogram_sizes = torch.tensor([[w, h, d]]).to(device)

    group_start_indices = [0]

    # Automatically determine num_select based on the number of unique classes
    # and the model's num_select per class.
    unique_class_num = len(set(prompts_label))
    if num_select is None:
        num_select = processor_3d.num_select * unique_class_num

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(
                all_images,
                tilt_angles=all_tilts,
                group_start_indices=group_start_indices,
                prompts=all_bboxes,
                categories=all_categories,
                tomogram_sizes=tomogram_sizes,
            )
            results_3d = processor_3d(
                outputs,
                tomogram_sizes,
                num_select=num_select,
                nms_threshold=nms_threshold,
            )

    estimated_centers = results_3d[0]['boxes_3d'].cpu().numpy()[:, :3]
    estimated_diameters = results_3d[0]['boxes_3d'].cpu().numpy()[:, 3:].mean(axis=1)
    estimated_labels = results_3d[0]['labels'].cpu().numpy()
    confidence_scores = results_3d[0]['scores'].cpu().numpy()

    if isinstance(confidence_score_threshold, float):
        if confidence_score_threshold <= 0.0:
            return estimated_centers, estimated_diameters, estimated_labels, confidence_scores

        confidence_score_threshold = {
            label: confidence_score_threshold for label in set(prompts_label)
        }

    keep_indices = []
    for idx, (label, score) in enumerate(zip(estimated_labels, confidence_scores)):
        threshold = confidence_score_threshold.get(label, 0.0)
        if score >= threshold:
            keep_indices.append(idx)

    keep_indices = np.array(keep_indices)
    estimated_centers = estimated_centers[keep_indices]
    estimated_diameters = estimated_diameters[keep_indices]
    estimated_labels = estimated_labels[keep_indices]
    confidence_scores = confidence_scores[keep_indices]

    return estimated_centers, estimated_diameters, estimated_labels, confidence_scores


def pipeline_get_prototypes(
    model: torch.nn.Module,
    tomogram_whd: t.Tuple[int, int, int],
    tiltseries: npt.NDArray[float],
    thetas: t.List[float],
    prompts_xyz: npt.NDArray[float],
    prompts_label: t.List[int],
    diameters: npt.NDArray[float],
    device: str,
) -> t.Dict[str, t.Union[torch.Tensor, t.Dict]]:
    """
    Executes the first part of the inference pipeline to obtain prototype embeddings from the model
    based on the input 3D coordinate prompts and tilt-series.
    Args:
        model (torch.nn.Module): The initialized 2D-to-3D detection model.
        tomogram_whd (Tuple[int, int, int]): The dimensions of the target tomogram (W, H, D).
        tiltseries (npt.NDArray[float]): The input tilt-series images of shape (C, H, W).
        thetas (List[float]): A list of tilt angles corresponding to each image channel in the
            tilt-series.
        prompts_xyz (npt.NDArray[float]): Initial 3D point prompts of shape (N, 3) in [x, y, z]
            format.
        prompts_label (List[int]): Class labels corresponding to the point prompts.
        diameters (npt.NDArray[float]): Initial estimated diameters for the prompts.
        device (str): The device for model computation ('cuda' or 'cpu').
    Returns:
        Dict[str, Union[torch.Tensor, Dict]]: A dictionary containing the prototype embeddings and
            related information to be used for the second part of the inference pipeline.
    """
    verify_inputs(
        tomogram_whd=tomogram_whd,
        tiltseries=tiltseries,
        thetas=thetas,
        prompts_xyz=prompts_xyz,
        prompts_label=prompts_label,
        diameters=diameters,
        device=device,
        confidence_score_threshold=0.0,
    )
    w, h, d = tomogram_whd
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    channels = tiltseries.shape[0]
    all_images = []
    all_bboxes = []
    all_categories = []
    all_tilts = []
    for theta, channel_idx in zip(thetas, list(range(channels))):
        image = tiltseries[channel_idx]

        # NOTE: Assuming input tilt-series is in a range that can be safely converted to uint8
        # without clipping.
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        rotated_xyz_indices = rotate_coordinate(xyz_indices=prompts_xyz, theta=theta, W=w, D=d)
        rotated_bboxes = xyz_to_xy_bbox(rotated_xyz_indices, diameters)

        image, bboxes = normalize(image, {'boxes': torch.tensor(rotated_bboxes)})
        all_images.append(image)
        all_bboxes.append(bboxes['boxes'])
        all_categories.append(prompts_label)
        all_tilts.append(theta)

    all_images = misc.nested_tensor_from_tensor_list(all_images)

    all_images = all_images.to(device)
    all_tilts = torch.tensor(all_tilts).to(device)
    all_bboxes = [bboxes.to(device) for bboxes in all_bboxes]
    all_categories = [torch.tensor(categories).to(device) for categories in all_categories]
    tomogram_sizes = torch.tensor([[w, h, d]]).to(device)

    group_start_indices = [0]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            prototypes_dict = model.forward_get_prototypes(
                all_images,
                tilt_angles=all_tilts,
                group_start_indices=group_start_indices,
                prompts=all_bboxes,
                categories=all_categories,
                tomogram_sizes=tomogram_sizes,
            )

    return prototypes_dict


def pipeline_inference_with_prototypes(
    model: torch.nn.Module,
    processor_3d: torch.nn.Module,
    tomogram_whd: t.Tuple[int, int, int],
    tiltseries: npt.NDArray[float],
    thetas: t.List[float],
    prototypes_dict: t.Dict[str, t.Union[torch.Tensor, t.Dict]],
    device: str,
    num_select: t.Optional[int] = None,
    nms_threshold: float = -1.0,
    confidence_score_threshold: t.Union[float, t.Dict[int, float]] = 0.0,
) -> t.Dict[str, t.Union[torch.Tensor, t.Dict]]:
    """
    Executes the second part of the inference pipeline using the prototype embeddings obtained from
    the first part to predict final 3D centers, diameters, labels, and confidence scores.
    Args:
        model (torch.nn.Module): The initialized 2D-to-3D detection model.
        processor_3d (torch.nn.Module): The 3D post-processing module.
        tomogram_whd (Tuple[int, int, int]): The dimensions of the target tomogram (W, H, D).
        tiltseries (npt.NDArray[float]): The input tilt-series images of shape (C, H, W).
        thetas (List[float]): A list of tilt angles corresponding to each image channel in the
            tilt-series.
        prototypes_dict (Dict[str, Union[torch.Tensor, Dict]]): The dictionary containing prototype
            embeddings and related information obtained from the first part of the inference
            pipeline.
        device (str): The device for model computation ('cuda' or 'cpu').
        num_select (Optional[int]): The number of top estimates to select for post-processing. If
            None, defaults to model's num_select multiplied by the number of unique classes in the
            prototypes.
        nms_threshold (float): The threshold for non-maximum suppression. If lower than 0, no NMS
            will be applied. Higher values result in more aggressive suppression. See the
            PostProcess3D class for details.
        confidence_score_threshold (Union[float, Dict[int, float]]): The confidence score threshold
            for filtering predictions. Can be a single float applied to all classes or a dictionary
            mapping class IDs to thresholds.
    Returns:
        tuple: A tuple containing the inference results:
            - estimated_centers (npt.NDArray[float]): Estimated 3D coordinates [x, y, z] of the
                detected objects.
            - estimated_diameters (npt.NDArray[float]): Estimated average 3D diameters.
            - estimated_labels (npt.NDArray[int]): Estimated class labels.
            - confidence_scores (npt.NDArray[float]): Confidence scores for the estimations.
    """
    verify_inputs(
        tomogram_whd=tomogram_whd,
        tiltseries=tiltseries,
        thetas=thetas,
        prompts_xyz=None,
        prompts_label=None,
        diameters=None,
        device=device,
        confidence_score_threshold=confidence_score_threshold,
        is_inference_with_prototypes=True,
    )
    w, h, d = tomogram_whd
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    channels = tiltseries.shape[0]
    all_images = []
    all_tilts = []
    for theta, channel_idx in zip(thetas, list(range(channels))):
        image = tiltseries[channel_idx]

        # NOTE: Assuming input tilt-series is in a range that can be safely converted to uint8
        # without clipping.
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image, _ = normalize(image, None)
        all_images.append(image)
        all_tilts.append(theta)

    all_images = misc.nested_tensor_from_tensor_list(all_images)

    all_images = all_images.to(device)
    all_tilts = torch.tensor(all_tilts).to(device)
    tomogram_sizes = torch.tensor([[w, h, d]]).to(device)

    group_start_indices = [0]

    # Automatically determine num_select based on the number of unique classes
    # and the model's num_select per class.
    prompts_label = prototypes_dict[
        'group_prompt_dict']['external_category_id_to_internal_category_id_maps'][0].keys()
    unique_class_num = len(set(prompts_label))
    if num_select is None:
        num_select = processor_3d.num_select * unique_class_num

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model.forward_with_prototypes(
                all_images,
                tilt_angles=all_tilts,
                group_start_indices=group_start_indices,
                tomogram_sizes=tomogram_sizes,
                prompt_embed_dict=prototypes_dict,
            )
            results_3d = processor_3d(
                outputs,
                tomogram_sizes,
                num_select=num_select,
                nms_threshold=nms_threshold,
            )

    estimated_centers = results_3d[0]['boxes_3d'].cpu().numpy()[:, :3]
    estimated_diameters = results_3d[0]['boxes_3d'].cpu().numpy()[:, 3:].mean(axis=1)
    estimated_labels = results_3d[0]['labels'].cpu().numpy()
    confidence_scores = results_3d[0]['scores'].cpu().numpy()

    if isinstance(confidence_score_threshold, float):
        if confidence_score_threshold <= 0.0:
            return estimated_centers, estimated_diameters, estimated_labels, confidence_scores

        confidence_score_threshold = {
            label: confidence_score_threshold for label in set(prompts_label)
        }

    keep_indices = []
    for idx, (label, score) in enumerate(zip(estimated_labels, confidence_scores)):
        threshold = confidence_score_threshold.get(label, 0.0)
        if score >= threshold:
            keep_indices.append(idx)

    keep_indices = np.array(keep_indices)
    estimated_centers = estimated_centers[keep_indices]
    estimated_diameters = estimated_diameters[keep_indices]
    estimated_labels = estimated_labels[keep_indices]
    confidence_scores = confidence_scores[keep_indices]

    return estimated_centers, estimated_diameters, estimated_labels, confidence_scores
