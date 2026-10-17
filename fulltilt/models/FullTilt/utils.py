# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_clones(module: nn.Module, N: int, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def gen_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    learnedwh=None
):
    """
    Input:
        - memory: bs, sum{hw}, d_model
        - memory_padding_mask: bs, sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, sum{hw}, d_model
        - output_proposals: bs, sum{hw}, 4
    """
    N_, _, _ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
        -1, keepdim=True
    )
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    # TODO: do we need this (.mean(1))?
    # return loss.mean(1).sum() / num_boxes
    return loss.sum() / num_boxes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'prelu':
        return nn.PReLU()
    if activation == 'selu':
        return F.selu

    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(pos_tensor.size(-1)))
    return pos


class ContrastiveEmbed(nn.Module):
    def __init__(self, max_unique_classes=256):
        super().__init__()
        self.max_unique_classes = max_unique_classes

    def forward(self, x, prompt_dict):
        assert isinstance(prompt_dict, dict)

        y = prompt_dict['encoded_prompt']
        prompt_token_mask = prompt_dict['prompt_token_mask']

        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~prompt_token_mask[:, None, :], float('-inf'))

        new_res = torch.full(
            (*res.shape[:-1], self.max_unique_classes), float('-inf'), device=res.device,
        )
        new_res[..., : res.shape[-1]] = res

        return new_res


def gen_sineembed_for_position_3d(pos_tensor):
    """
    Args:
        pos_tensor: [**, 5]
                    containing (cx, cy, cz, w, h) normalized in [0, 1]
    Returns:
        pos: [**, 640]
             (128*5 = 640 dims)
    """
    scale = 2 * math.pi

    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)

    x_embed = pos_tensor[:, :, 0] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    y_embed = pos_tensor[:, :, 1] * scale
    pos_y = y_embed[:, :, None] / dim_t
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    z_embed = pos_tensor[:, :, 2] * scale
    pos_z = z_embed[:, :, None] / dim_t
    pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

    w_embed = pos_tensor[:, :, 3] * scale
    pos_w = w_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    h_embed = pos_tensor[:, :, 4] * scale
    pos_h = h_embed[:, :, None] / dim_t
    pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_y, pos_z, pos_w, pos_h), dim=2)

    return pos


def project_3d_to_2d_batch(reference_points_3d, tilt_angles_flatten, dimensions_3d):
    """
    Args:
        reference_points_3d: [B_total, Nq, 5] (cx, cy, cz, w, h) normalized [0,1]
        tilt_angles_flatten: [B_total]
        dimensions_3d: [B_total, 3] containing (W, H, D) for each image
    """
    device = reference_points_3d.device
    W = dimensions_3d[:, 0:1]
    H = dimensions_3d[:, 1:2]
    D = dimensions_3d[:, 2:3]

    scale_factor = torch.cat([W, H, D, W, H], dim=1)
    points_physical = reference_points_3d * scale_factor.unsqueeze(1)

    x = points_physical[..., 0] - W / 2
    y = points_physical[..., 1] - H / 2
    z = points_physical[..., 2] - D / 2
    w_phy = points_physical[..., 3]
    h_phy = points_physical[..., 4]

    thetas = torch.deg2rad(tilt_angles_flatten).to(device)
    cos_t = torch.cos(thetas).unsqueeze(1)
    sin_t = torch.sin(thetas).unsqueeze(1)

    x_proj = x * cos_t + z * sin_t
    y_proj = y

    x_2d = x_proj + W / 2
    y_2d = y_proj + H / 2

    x_2d = x_2d / W
    y_2d = y_2d / H
    w_out = w_phy / W
    h_out = h_phy / H

    return torch.stack([x_2d, y_2d, w_out, h_out], dim=-1)
