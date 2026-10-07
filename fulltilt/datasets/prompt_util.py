# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import copy
import random
import typing as t

import torch

from fulltilt.util.misc import nested_tensor_from_tensor_list


def prepare_prompts_aggressive(
    targets: t.List[t.Dict],
    labels2indices_list: t.List[t.Dict[int, t.List[int]]],
    labels2count_list: t.List[t.Dict[int, int]],
    max_pick_prompt_num: int = 5,
    partial_pick_label: int = -1,  # partically pick some labels; -1 = pick all
    drop_tilt_prompt_max_ratio: float = 0.0,  # drop tilts' prompts
) -> t.List[t.Dict]:
    image_num = len(targets)
    if len(labels2indices_list) != image_num:
        raise ValueError('len(labels2indices_list) should be equal to image_num')
    if len(labels2count_list) != image_num:
        raise ValueError('len(labels2count_list) should be equal to image_num')

    for i in range(image_num - 1):
        if labels2indices_list[i] != labels2indices_list[i + 1]:
            raise ValueError(
                'labels2indices_list[i] should be equal to labels2indices_list[i + 1] for all i'
            )
        if labels2count_list[i] != labels2count_list[i + 1]:
            raise ValueError(
                'labels2count_list[i] should be equal to labels2count_list[i + 1] for all i'
            )

    # generate prompts
    all_labels = labels2indices_list[0].keys()
    # pick >= 1 labels
    if partial_pick_label != -1:
        pick_label_num = random.randint(min(partial_pick_label, len(all_labels)), len(all_labels))
    else:
        pick_label_num = random.randint(len(all_labels), len(all_labels))
    picked_labels = random.sample(list(all_labels), k=pick_label_num)

    picked_indices = []
    keep = []

    # NOTE: we assume prompts in each tilt has the same order
    for picked_label in picked_labels:
        label_indices = labels2indices_list[0][picked_label]
        label_count = labels2count_list[0][picked_label]
        pick_num = random.randint(
            1,
            min(max_pick_prompt_num, label_count),  # TODO
        )
        picked_label_indices = random.sample(list(range(label_count)), k=pick_num)
        for picked_label_index in picked_label_indices:
            picked_indices.append(label_indices[picked_label_index])
        keep += label_indices

    # picked_indices = sorted(picked_indices)  # TODO: can shuffle?
    random.shuffle(picked_indices)
    keep.sort()

    if drop_tilt_prompt_max_ratio > 0:
        drop_tilt_prompt_ratio = random.uniform(0.0, drop_tilt_prompt_max_ratio)
        drop_tilt_prompt_num = int(drop_tilt_prompt_ratio * image_num)
        # Clamp to ensure we don't drop more than exists (unlikely but safe)
        drop_tilt_prompt_num = min(drop_tilt_prompt_num, image_num)
    else:
        drop_tilt_prompt_num = 0
    drop_target_in_tilts = set(random.sample(list(range(image_num)), k=drop_tilt_prompt_num))

    prompts = []
    for target_idx, target in enumerate(targets):
        if target_idx in drop_target_in_tilts:
            sub_prompts = {'boxes': [], 'labels': []}
        else:
            sub_prompts = {
                'boxes': target['boxes'][picked_indices],
                'labels': target['labels'][picked_indices],
            }
        prompts.append(sub_prompts)

    targets_copy = copy.deepcopy(targets)
    for target in targets_copy:
        target['boxes'] = target['boxes'][keep]
        target['labels'] = target['labels'][keep]
        target['area'] = target['area'][keep]
        target['iscrowd'] = target['iscrowd'][keep]

    return prompts, targets_copy, keep


def prepare_prompts(
    targets: t.List[t.Dict],
    labels2indices: t.Dict[int, t.List[int]],
    labels2count: t.Dict[int, int],
    max_pick_num: int = 5,
) -> t.List[t.Dict]:
    """
    Randomly sample a subset of object instances across grouped images to form prompt annotations.

    This function selects specific instances (labels and boxes) across a sequence of related
    images (e.g., different tilts or time steps) to be used as prompts. It ensures that
    if an instance is picked in one image, the corresponding instance in other related
    images is also picked.

    Parameters
    ----------
    targets : List[Dict[str, torch.Tensor]]
        List of target dicts per image. Each dict must have:
        - 'boxes': Tensor of shape (N, 4)
        - 'labels': Tensor of shape (N,)
    labels2indices : Dict[int, List[int]]
        Maps a label ID to a list of global indices (flattened across all images).
    labels2count : Dict[int, int]
        Maps a label ID to the number of unique instances of that label per image.
    max_pick_num : int, optional
        The maximum number of unique instances to pick for each selected label.
        Default is 5.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        A list of prompt dictionaries per image. Each dictionary contains:
        - 'boxes': Tensor of shape (M, 4) or empty list if no matches.
        - 'labels': Tensor of shape (M,) or empty list if no matches.

    Raises
    ------
    ValueError
        If the number of global indices is not a perfect multiple of label_count,
        or if an index calculation falls outside of the current image range.
    """

    # generate prompts
    all_labels = labels2indices.keys()
    # pick >= 1 labels
    # TODO: sample 1~all labels (need to filter out images w/o any labels)
    pick_num = random.randint(len(all_labels), len(all_labels))
    picked_labels = random.sample(list(all_labels), k=pick_num)

    picked_indices = []

    # NOTE: we assume prompts in each tilt has the same order
    for picked_label in picked_labels:
        label_indices = labels2indices[picked_label]
        label_count = labels2count[picked_label]
        pick_num = random.randint(
            1,
            min(max_pick_num, label_count),  # TODO
        )
        picked_label_indices = random.sample(list(range(label_count)), k=pick_num)

        # Calculate how many images these instances appear in
        if len(label_indices) % label_count != 0:
            raise ValueError(
                f'Total indices for label {picked_label} must be a multiple of label_count.'
            )

        image_num = len(label_indices) // label_count  # should be int

        # pick the same particles in all tilts
        for curr_image_num in range(image_num):
            for picked_label_index in picked_label_indices:
                picked_indices.append(
                    label_indices[curr_image_num * label_count + picked_label_index]
                )

    picked_indices = sorted(picked_indices)

    prompts = []
    current_index = 0
    picked_indices_pointer = 0
    for target in targets:
        sub_prompts = {'boxes': [], 'labels': []}
        min_range = current_index
        max_range = current_index + len(target['boxes'])
        while (
            picked_indices_pointer < len(picked_indices)
            and picked_indices[picked_indices_pointer] < max_range
        ):
            if picked_indices[picked_indices_pointer] < min_range:
                raise ValueError(
                    f'Min range is {min_range} but pick {picked_indices[picked_indices_pointer]}'
                )
            sub_prompts['boxes'].append(
                target['boxes'][picked_indices[picked_indices_pointer] - min_range]
            )
            sub_prompts['labels'].append(
                target['labels'][picked_indices[picked_indices_pointer] - min_range]
            )
            picked_indices_pointer += 1
        if len(sub_prompts['boxes']) > 0:
            sub_prompts['boxes'] = torch.stack(sub_prompts['boxes'])
            sub_prompts['labels'] = torch.stack(sub_prompts['labels'])
        prompts.append(sub_prompts)
        current_index += len(target['boxes'])

    return prompts


def shuffle_data(
    imgs: t.List[torch.Tensor],
    tilts: torch.Tensor,
    targets: t.List[t.Dict[str, t.Any]],
    prompts: t.List[t.Dict],
    random_order_aug: bool = False,
    subsample_val: bool = False,
    max_training_tilts: int = 31,
    min_training_tilts: int = 21,
    max_eval_tilts: int = 31,
):
    if not random_order_aug:
        if subsample_val:
            if len(imgs) > max_eval_tilts:
                if len(imgs) == 41:
                    selected_idx = list(range(10, 31))
                else:
                    selected_idx = list(range(0, len(imgs), 2))
                imgs_sub = [imgs[i] for i in selected_idx]
                tilts_sub = [tilts[i] for i in selected_idx]
                targets_sub = [targets[i] for i in selected_idx]
                prompts_sub = [prompts[i] for i in selected_idx]
                return imgs_sub, tilts_sub, targets_sub, prompts_sub
            else:
                return imgs, tilts, targets, prompts
        else:
            return imgs, tilts, targets, prompts

    # 1. Separate indices based on content
    useful_idx = [i for i, p in enumerate(prompts) if len(p.get('boxes', [])) > 0]
    # empty_idx = [i for i, p in enumerate(prompts) if len(p.get('boxes', [])) == 0]

    all_indices = list(range(len(imgs)))

    # 2. Determine how many samples to take
    n = min(len(all_indices), max_training_tilts)
    num = random.randint(max(min(min_training_tilts, n), n // 2), n)

    selected_idx = []

    # 3. Priority: Pick at least one useful index if any exist
    if useful_idx:
        primary = random.choice(useful_idx)
        selected_idx.append(primary)
        # Remove it from the pool so we don't pick it twice
        all_indices.remove(primary)

    # 4. Fill the remaining 'num' slots from the rest of the data
    remaining_count = num - len(selected_idx)
    if remaining_count > 0:
        selected_idx.extend(random.sample(all_indices, remaining_count))

    # 5. Final shuffle so the 'useful' one isn't always at the start
    random.shuffle(selected_idx)

    # Map back to data
    imgs_shuf = [imgs[i] for i in selected_idx]
    tilts_shuf = [tilts[i] for i in selected_idx]
    targets_shuf = [targets[i] for i in selected_idx]
    prompts_shuf = [prompts[i] for i in selected_idx]

    return imgs_shuf, tilts_shuf, targets_shuf, prompts_shuf


def check_prompt_valid(prompts: t.List[t.Dict]) -> bool:
    is_valid = False
    for prompt in prompts:
        if prompt['boxes'] != [] and prompt['labels'] != []:
            is_valid = True
            break
    return is_valid


class VisualPromptEvaluator:
    def __init__(
        self,
        imgs: t.List[torch.Tensor],
        tilts: torch.Tensor,
        targets: t.List[t.Dict[str, t.Any]],
        gt_3d_torch: t.Dict[str, torch.Tensor],
    ):
        self.imgs = imgs
        self.tilts = tilts
        self.targets = targets
        self.gt_3d_torch = gt_3d_torch

        # Reference labels from the first tilt (consistent across group)
        self.label_to_indices = {}
        for idx, label in enumerate(self.targets[0]['labels'].tolist()):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def get_label_stats(self) -> dict[int, int]:
        return {label: len(indices) for label, indices in self.label_to_indices.items()}

    def get_image_num(self):
        return len(self.imgs)

    def has_label(self, label_id: int) -> bool:
        return label_id in self.label_to_indices

    def has_labels(self, label_ids: list[int]) -> bool:
        for label_id in label_ids:
            if label_id not in self.label_to_indices:
                return False
        return True

    def pack_images(
        self,
        label_ids: list[int],
        image_indices: t.Optional[list[int]] = None,
    ) -> t.Dict[str, t.Any]:
        if image_indices is None:
            image_indices = list(range(self.get_image_num()))
        out_imgs = [self.imgs[i] for i in image_indices]
        out_tilts = self.tilts[image_indices]

        label_set = set(label_ids)
        gt_3d_filtered = copy.deepcopy(self.gt_3d_torch)
        gt_labels = gt_3d_filtered['labels']

        # Create boolean mask for 3D particles matching selected labels
        gt_keep_mask = torch.tensor([label.item() in label_set for label in gt_labels])

        gt_3d_filtered['boxes_3d'] = gt_3d_filtered['boxes_3d'][gt_keep_mask]
        gt_3d_filtered['labels'] = gt_3d_filtered['labels'][gt_keep_mask]
        return {
            'imgs': nested_tensor_from_tensor_list(out_imgs),
            'tilts': [out_tilts],
            'gt_3d': [gt_3d_filtered],
        }

    def random_pick(
        self,
        label_ids: list[int],
        prompt_num: int,
        prompt_image_indices=None,
        image_indices=None,
    ):
        label_stats = self.get_label_stats()
        prompt_indices_per_label = []
        for label_id in label_ids:
            if label_id not in label_stats:
                raise ValueError(f'label_ids {label_id} is invalid')
            max_num = label_stats[label_id]
            if prompt_num > max_num:
                raise ValueError(
                    f'prompt_num {prompt_num} is over max_num of {max_num} for label {label_id}'
                )
            selected_prompt_ids = random.sample(range(max_num), prompt_num)
            prompt_indices_per_label.append(selected_prompt_ids)

        if prompt_image_indices is None:
            prompt_image_indices = list(range(self.get_image_num()))

        if image_indices is None:
            image_indices = list(range(self.get_image_num()))

        return self.format_inference_data(
            image_indices=image_indices,
            label_ids=label_ids,
            prompt_indices_per_label=prompt_indices_per_label,
            prompt_image_indices=prompt_image_indices,
        )

    def format_inference_data(
        self,
        image_indices: list[int],
        label_ids: list[int],
        prompt_indices_per_label: list[list[int]],
        prompt_image_indices: list[int]
    ):
        """
        Args:
            image_indices: Tilts to include (e.g., [0, 10, 20])
            label_ids: Categories to evaluate (e.g., [1, 3])
            prompt_indices_per_label: Which instances of those categories act as prompts
            prompt_image_indices: Which images (from image_indices) show the prompts
        """
        if not set(prompt_image_indices).issubset(set(image_indices)):
            raise ValueError('prompt_image_indices should be a subset of image_indices.')

        # 1. Map requested prompts to global annotation indices
        final_prompt_indices = []
        for i, label in enumerate(label_ids):
            available_indices = self.label_to_indices.get(label, None)
            if available_indices is None:
                raise ValueError('Label does not exist')
            requested_sub_indices = prompt_indices_per_label[i]
            for sub_idx in requested_sub_indices:
                if sub_idx < len(available_indices):
                    final_prompt_indices.append(available_indices[sub_idx])
                else:
                    raise ValueError('Indices for label out of range.')

        # 2. Select Images and Tilts
        out_imgs = [self.imgs[i] for i in image_indices]
        out_tilts = self.tilts[image_indices]

        # 3. Filter 2D Targets and Generate Prompts
        # We only keep the categories specified in label_ids for the Ground Truth
        out_targets_2d = []
        prompts = []

        # Pre-calculate a mask for the 2D targets based on label_ids
        # This ensures evaluation only looks at selected categories
        label_set = set(label_ids)

        for idx in image_indices:
            # Filtering the 2D GT for this tilt
            target = self.targets[idx]
            keep_mask = torch.tensor([label.item() in label_set for label in target['labels']])

            filtered_target = {
                'boxes': target['boxes'][keep_mask],
                'labels': target['labels'][keep_mask],
                'image_id': target['image_id']
            }
            out_targets_2d.append(filtered_target)

            # Assigning prompts
            if idx in prompt_image_indices:
                sub_prompts = {
                    'boxes': target['boxes'][final_prompt_indices],
                    'labels': target['labels'][final_prompt_indices],
                }
            else:
                sub_prompts = {
                    'boxes': [],
                    'labels': [],
                }
            prompts.append(sub_prompts)

        # 4. Filter 3D Ground Truth
        # Only keep boxes/labels in gt_3d_torch that match label_ids
        gt_3d_filtered = copy.deepcopy(self.gt_3d_torch)
        gt_labels = gt_3d_filtered['labels']

        # Create boolean mask for 3D particles matching selected labels
        gt_keep_mask = torch.tensor([label.item() in label_set for label in gt_labels])

        gt_3d_filtered['boxes_3d'] = gt_3d_filtered['boxes_3d'][gt_keep_mask]
        gt_3d_filtered['labels'] = gt_3d_filtered['labels'][gt_keep_mask]
        # tomogram_size remains the same

        return {
            'imgs': nested_tensor_from_tensor_list(out_imgs),
            'tilts': [out_tilts],
            'prompts': [prompts],
            'targets_2d': [out_targets_2d],
            'gt_3d': [gt_3d_filtered],
        }
