# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import typing as t

import torch

import fulltilt.util.misc as misc


def prepare_for_cdn(
    dn_args: t.Tuple[t.Dict, int, int, float],
    training: bool,
    num_queries: int,
    hidden_dim: int,
    label_enc: torch.Tensor,
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, t.Dict]:
    """
    3D CDN for FullTilt.
    Args:
        dn_args: Tuple containing (targets, dn_number, _, box_noise_scale)
        training: Whether in training mode (DN is only for training)
        num_queries: Number of original queries (without DN)
        hidden_dim: Dimension of label embeddings
        label_enc: Precomputed label embeddings [num_classes, hidden_dim]
    Returns:
        input_query_label: [B, Total_DN, hidden_dim] Label embeddings for DN queries
        input_query_bbox: [B, Total_DN, 5] Box embeddings for DN queries (5D: x, y, z, w, h)
        attn_mask: [Total_DN + num_queries, Total_DN + num_queries] Attention mask for DN queries
        dn_meta: Dictionary containing DN metadata (e.g., valid_mask, positive_mask, scalar)
    """
    if not training:
        return None, None, None, None

    targets, dn_number, _, box_noise_scale = dn_args
    if dn_number == 0:
        return None, None, None, None

    # TODO: elegant way to get device
    device = targets[0]['labels'].device

    # ----------------------------------------------------
    # 1. Calculate # of DN Group
    # ----------------------------------------------------
    # positive and negative dn queries
    dn_number = dn_number * 2
    known = [(torch.ones_like(t['labels'])) for t in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]

    # Calculate scalar (how many times to repeat GT)
    if int(max(known_num)) == 0:
        dn_number = 1
    else:
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
    if dn_number == 0:
        dn_number = 1

    # ----------------------------------------------------
    # 2. Prepare GT Tensors (before exetension)
    # ----------------------------------------------------
    # [N_total]
    labels = torch.cat([t['labels'] for t in targets])
    # [N_total, 5] (x, y, z, w, h)
    boxes = torch.cat([t['boxes_3d'] for t in targets])
    # [N_total]
    batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

    # ----------------------------------------------------
    # 3. Extend (Repeat) to be DN Queries
    # ----------------------------------------------------
    # Repeat 2 * dn_number times
    # known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
    known_bboxs = boxes.repeat(2 * dn_number, 1)  # [Total_DN, 5]

    known_labels_expaned = known_labels.clone()
    known_bbox_expand = known_bboxs.clone()

    # ----------------------------------------------------
    # 5. Box Noise (For 5D)
    # ----------------------------------------------------
    single_pad = int(max(known_num))
    pad_size = int(single_pad * 2 * dn_number)

    # Mark Positive / Negative Indices
    # half part Positive Group; later part: Negative Group
    total_boxes_len = len(boxes)
    positive_idx = torch.tensor(
        range(total_boxes_len)
    ).long().to(device).unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (
        torch.tensor(range(dn_number)) * total_boxes_len * 2
    ).long().to(device).unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + total_boxes_len  # Negative is shifted by one GT set

    if box_noise_scale > 0:
        # 5D Noise Injection Strategy
        # XY: Corner Noise (DINO Style)
        # Z: Center Noise (Simple Additive)

        diff = torch.zeros_like(known_bboxs)

        # W, H are at indices 3, 4
        w = known_bboxs[:, 3]
        h = known_bboxs[:, 4]

        # Diff def (Noise Scale Base)
        diff[:, 0] = w / 2  # for x
        diff[:, 1] = h / 2  # for y
        diff[:, 2] = (w + h) / 4  # for z (heuristic)
        diff[:, 3] = w / 2  # for w (corner conversion logic)
        diff[:, 4] = h / 2  # for h

        # Random Noise
        rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(known_bboxs)

        # Negative samples get larger noise (+1.0 shift)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign

        # Apply Noise
        noise = torch.mul(rand_part, diff).to(device) * box_noise_scale

        # Update Boxes
        new_box = known_bboxs.clone()

        # X Axis
        # noise[:, 0] is nosie for x1, noise[:, 3] is noise for x2
        new_x1 = (known_bboxs[:, 0] - w / 2) + noise[:, 0]
        new_x2 = (known_bboxs[:, 0] + w / 2) + noise[:, 3]
        new_box[:, 0] = (new_x1 + new_x2) / 2
        new_box[:, 3] = (new_x2 - new_x1).abs()  # W should be positive

        # Y Axis
        # noise[:, 1] is noise for y1, noise[:, 4] is noise for y2
        new_y1 = (known_bboxs[:, 1] - h / 2) + noise[:, 1]
        new_y2 = (known_bboxs[:, 1] + h / 2) + noise[:, 4]
        new_box[:, 1] = (new_y1 + new_y2) / 2
        new_box[:, 4] = (new_y2 - new_y1).abs()

        # Z Axis (Simple Center Noise)
        new_box[:, 2] = known_bboxs[:, 2] + noise[:, 2]

        # Clamp & Assign
        known_bbox_expand = new_box.clamp(min=0.0, max=1.0)

    # ----------------------------------------------------
    # 6. Embedding & Padding
    # ----------------------------------------------------
    m = known_labels_expaned.long().to(device)

    # label_enc: [B, Seq, C]
    # known_bid: [N_dn]
    # m: [N_dn]
    input_label_embed = label_enc[known_bid, m]

    input_bbox_embed = misc.inverse_sigmoid(known_bbox_expand)

    # Prepare Output Buffer (Padding)
    padding_label = torch.zeros(pad_size, hidden_dim).to(device)
    padding_bbox = torch.zeros(pad_size, 5).to(device)  # 5D

    input_query_label = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    # Valid Mask (Makr non-padding region)
    output_valid_mask = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=device)
    output_positive_mask = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=device)

    map_known_indice = torch.tensor([]).to(device)
    if len(known_num):
        map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
        map_known_indice = torch.cat(
            [map_known_indice + single_pad * i for i in range(2 * dn_number)]
        ).long()

    if len(known_bid):
        input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # Fill Masks
        output_valid_mask[(known_bid.long(), map_known_indice)] = True

        # Fill Positive Mask
        group_idx = map_known_indice // single_pad
        is_pos = (group_idx % 2 == 0)
        output_positive_mask[(known_bid.long(), map_known_indice)] = is_pos.to(device)

    # ----------------------------------------------------
    # 7. Attention Mask (Block Diagonal)
    # ----------------------------------------------------
    tgt_size = pad_size + num_queries
    attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0

    # Match query cannot see the reconstruct (DN)
    attn_mask[pad_size:, :pad_size] = True

    # Reconstruct cannot see each other (Inter-Group Masking)
    for i in range(dn_number):
        # i is Pos+Neg Pair
        # Group Start/End
        start = single_pad * 2 * i
        end = single_pad * 2 * (i + 1)

        # Mask other regions
        if i == 0:
            attn_mask[start:end, end:pad_size] = True
        elif i == dn_number - 1:
            attn_mask[start:end, :start] = True
        else:
            attn_mask[start:end, end:pad_size] = True
            attn_mask[start:end, :start] = True

    dn_meta = {
        'pad_dn': pad_size,
        'num_dn_group': dn_number,
        'valid_mask': output_valid_mask,
        'positive_mask': output_positive_mask,
        'targets': targets,
        'scalar': dn_number,
    }

    return input_query_label, input_query_bbox, attn_mask, dn_meta
