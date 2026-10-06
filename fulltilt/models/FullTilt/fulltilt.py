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
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

import fulltilt.models.FullTilt.backbone.backbone as fulltilt_backbone
import fulltilt.models.FullTilt.dn_components as dn_components
import fulltilt.models.FullTilt.matcher as fulltilt_matcher
import fulltilt.models.FullTilt.transformer as fulltilt_transformer
from fulltilt.models import registry
from fulltilt.models.FullTilt import utils
from fulltilt.util import box_ops, eval_3d, misc


class FullTilt(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        num_queries: int,
        query_dim: int = 5,
        num_feature_levels: int = 4,
        nheads: int = 8,
        max_unique_classes: int = 256,
        aux_loss: bool = False,
        dn_number: int = 100,
        dn_box_noise_scale: float = 0.4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_unique_classes = max_unique_classes
        self.query_dim = query_dim

        self.aux_loss = aux_loss
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale

        if num_feature_levels <= 1:
            raise NotImplementedError

        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.num_channels[_]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)

        _class_embed = utils.ContrastiveEmbed(max_unique_classes=max_unique_classes)
        _bbox_embed = utils.MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _bbox_embed3d = utils.MLP(hidden_dim, hidden_dim, self.query_dim, 3)
        nn.init.constant_(_bbox_embed3d.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed3d.layers[-1].bias.data, 0)

        self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
        self.transformer.enc_out_bbox_embed = _bbox_embed

        class_embed_layerlist = [_class_embed for _ in range(transformer.num_decoder_layers)]
        box_embed_layerlist = [_bbox_embed3d for _ in range(transformer.num_decoder_layers)]
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.bbox_embed3d = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed3d = self.bbox_embed3d

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: misc.NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = misc.nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self, 'poss'):
            del self.poss

    def check_input_and_process_samples(
        self,
        samples: t.Union[torch.Tensor, misc.NestedTensor],
        tilt_angles: torch.Tensor,
        prompts: t.Optional[t.List[t.Union[torch.Tensor, t.List]]],
        categories: t.Optional[t.List[t.Union[torch.Tensor, t.List]]],
        tomogram_sizes: torch.Tensor,
    ) -> misc.NestedTensor:
        if not isinstance(samples, misc.NestedTensor):
            samples = misc.nested_tensor_from_tensor_list(samples)

        if len(tilt_angles) != samples.tensors.shape[0]:
            raise ValueError('tilt_angles should have the same size of batch size')
        if prompts is not None:
            if len(prompts) != samples.tensors.shape[0]:
                raise ValueError('prompts should have the same size of batch size')
        if categories is not None:
            if len(categories) != samples.tensors.shape[0]:
                raise ValueError('categories should have the same size of batch size')
        if tomogram_sizes is None:
            raise ValueError('tomogram_sizes should be provided.')

        return samples

    def forward_backbone(
        self,
        samples: misc.NestedTensor,
    ) -> t.Tuple[t.List[torch.Tensor], t.List[torch.Tensor]]:
        srcs = []
        masks = []
        for level, feat in enumerate(self.features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
            if mask is None:
                raise ValueError(
                    f'Mask is None for level {level}, which is not expected. Please check the'
                    'backbone output.'
                )

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for level in range(_len_srcs, self.num_feature_levels):
                if level == _len_srcs:
                    src = self.input_proj[level](self.features[-1].tensors)
                else:
                    src = self.input_proj[level](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](misc.NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        return srcs, masks

    def forward(
        self,
        samples: t.Union[torch.Tensor, misc.NestedTensor],
        tilt_angles: torch.Tensor,
        group_start_indices: t.List[int],
        prompts: t.List[t.Union[torch.Tensor, t.List]],
        categories: t.List[t.Union[torch.Tensor, t.List]],
        tomogram_sizes: torch.Tensor,
        gt_3d_batch: t.Optional[t.List[t.Dict]] = None,
        force_aux_interm_loss: bool = False,
        **kw,
    ) -> t.Dict[str, t.Union[torch.Tensor, t.Dict]]:
        group_start_indices = group_start_indices.copy()
        samples = self.check_input_and_process_samples(
            samples, tilt_angles, prompts, categories, tomogram_sizes,
        )

        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs, masks = self.forward_backbone(samples)

        total_images = len(tilt_angles) if isinstance(tilt_angles, list) else tilt_angles.shape[0]
        indices = group_start_indices + [total_images]
        group_counts = torch.tensor(
            [indices[i + 1] - indices[i] for i in range(len(indices) - 1)],
            device=tilt_angles.device
        )

        encoder_outputs = self.transformer.forward_encoder(
            srcs=srcs,
            masks=masks,
            pos_embeds=self.poss,
            tilt_angles=tilt_angles,
            group_start_indices=group_start_indices,
            prompts=prompts,
            categories=categories,
        )

        if self.training:
            if gt_3d_batch is None:
                raise ValueError('gt_3d_batch should be provided for training')

            mapped_targets = []
            group_maps = encoder_outputs[
                'group_prompt_dict'
            ]['external_category_id_to_internal_category_id_maps']
            for gt_3d, mapping in zip(gt_3d_batch, group_maps):
                gt_3d_mapped = gt_3d.copy()
                raw_labels = gt_3d['labels']
                new_labels = torch.tensor(
                    [mapping[int(label.item())] for label in raw_labels],
                    device=gt_3d['labels'].device, dtype=raw_labels.dtype
                )
                gt_3d_mapped['labels'] = new_labels
                mapped_targets.append(gt_3d_mapped)

            input_query_label, input_query_bbox, attn_mask, dn_meta = dn_components.prepare_for_cdn(
                dn_args=(mapped_targets, self.dn_number, 0, self.dn_box_noise_scale),
                training=True,
                num_queries=self.num_queries,
                hidden_dim=self.hidden_dim,
                label_enc=encoder_outputs['group_prompt_dict']['encoded_prompt'],  # [1, L, C]
            )

        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        (
            prompt_dict, group_prompt_dict, hs, reference, hs_enc, ref_enc, _
        ) = self.transformer.forward_decoder(
            encoder_outputs=encoder_outputs,
            tilt_angles=tilt_angles,
            tomogram_sizes=tomogram_sizes,
            input_query_bbox=input_query_bbox,
            input_query_label=input_query_label,
            attn_mask=attn_mask,
        )

        outputs_coord_list = []
        for (layer_ref_sig, layer_bbox_embed, layer_hs) in zip(
            reference[:-1], self.bbox_embed3d, hs,
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + misc.inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, group_prompt_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        dn_outputs = None
        if dn_meta is not None:
            pad_dn = dn_meta['pad_dn']

            # seperate DN part (first pad_dn) [Layers, B_groups, DN, ...]
            dn_out_logits = outputs_class[:, :, :pad_dn, :]
            dn_out_boxes = outputs_coord_list[:, :, :pad_dn, :]

            # preserve DN Outputs (put into out later)
            dn_outputs = {
                'pred_logits': dn_out_logits,
                'pred_boxes': dn_out_boxes,
                'dn_meta': dn_meta,
            }

            # update Main Outputs (Only preserve Matching Queries)
            # Left [Layers, B_groups, Nq, ...] will be processed by the standard process
            outputs_class = outputs_class[:, :, pad_dn:, :]
            outputs_coord_list = outputs_coord_list[:, :, pad_dn:, :]

        logits_group = outputs_class[-1]
        final_logits = logits_group.repeat_interleave(group_counts, dim=0)

        boxes_3d_group = outputs_coord_list[-1]
        final_boxes_3d_expanded = boxes_3d_group.repeat_interleave(group_counts, dim=0)

        tomogram_sizes_expanded = tomogram_sizes.repeat_interleave(group_counts, dim=0)
        final_boxes_2d = utils.project_3d_to_2d_batch(
            final_boxes_3d_expanded,
            tilt_angles,
            tomogram_sizes_expanded,
        )

        out = {
            'pred_logits': final_logits,
            'pred_boxes': final_boxes_2d,
            'pred_logits_group': logits_group,
            'pred_boxes_3d': boxes_3d_group,
            'internal_category_id_to_external_category_id_maps': prompt_dict[
                'internal_category_id_to_external_category_id_maps'
            ],
            'external_category_id_to_internal_category_id_maps': prompt_dict[
                'external_category_id_to_internal_category_id_maps'
            ],
            'group_internal_map': group_prompt_dict[
                'internal_category_id_to_external_category_id_maps'
            ],
            'group_external_map': group_prompt_dict[
                'external_category_id_to_internal_category_id_maps'
            ],
        }

        if dn_outputs is not None:
            out['dn_outputs'] = dn_outputs

        out['prompt_token_mask'] = prompt_dict['prompt_token_mask']
        out['prompt_token_mask_group'] = group_prompt_dict['prompt_token_mask']

        if (self.training or force_aux_interm_loss) and self.aux_loss:
            out['aux_outputs'] = []
            for i in range(len(outputs_class) - 1):
                # Group Level
                aux_logits_group = outputs_class[i]
                aux_3d_group = outputs_coord_list[i]

                # Image Level Expand
                aux_logits = aux_logits_group.repeat_interleave(group_counts, dim=0)
                aux_3d_exp = aux_3d_group.repeat_interleave(group_counts, dim=0)
                aux_2d = utils.project_3d_to_2d_batch(
                    aux_3d_exp,
                    tilt_angles,
                    tomogram_sizes_expanded,
                )

                out['aux_outputs'].append({
                    'pred_logits': aux_logits,
                    'pred_boxes': aux_2d,
                    'pred_logits_group': aux_logits_group,
                    'pred_boxes_3d': aux_3d_group,
                    'prompt_token_mask': out['prompt_token_mask'],
                    'prompt_token_mask_group': out['prompt_token_mask_group'],
                    'internal_category_id_to_external_category_id_maps': prompt_dict[
                        'internal_category_id_to_external_category_id_maps'
                    ],
                    'external_category_id_to_internal_category_id_maps': prompt_dict[
                        'external_category_id_to_internal_category_id_maps'
                    ],
                    'group_internal_map': group_prompt_dict[
                        'internal_category_id_to_external_category_id_maps'
                    ],
                    'group_external_map': group_prompt_dict[
                        'external_category_id_to_internal_category_id_maps'
                    ],
                })

        if self.training or force_aux_interm_loss:
            # prepare intermediate outputs

            # Calculate Group Level Output
            # ref_enc: [1, B_groups, Nq, 5] -> [B_groups, Nq, 5]
            interm_coord_group = ref_enc[-1]

            # hs_enc: [1, B_groups, Nq, 256] -> [B_groups, Nq, 256]
            # [Fix] Must use group_prompt_dict as hs_enc is extended from group_memory (0 deg)
            interm_class_group = self.transformer.enc_out_class_embed(hs_enc[-1], group_prompt_dict)

            # Extended to Image Level
            # Expand Logits
            interm_class_expanded = interm_class_group.repeat_interleave(group_counts, dim=0)

            # Expand & Project Boxes (3D -> 2D)
            interm_3d_expanded = interm_coord_group.repeat_interleave(group_counts, dim=0)
            interm_boxes_2d = utils.project_3d_to_2d_batch(
                interm_3d_expanded,
                tilt_angles,
                tomogram_sizes_expanded,
            )

            out['interm_outputs'] = {
                # [Standard 2D Keys]
                'pred_logits': interm_class_expanded,
                'pred_boxes': interm_boxes_2d,
                'prompt_token_mask': out['prompt_token_mask'],  # Image Level Mask

                # for loss_bbox_3d_interm
                'pred_logits_group': interm_class_group,
                'pred_boxes_3d': interm_coord_group,
                'prompt_token_mask_group': out['prompt_token_mask_group'],  # Group Level Mask

                # ID Maps
                'internal_category_id_to_external_category_id_maps': prompt_dict[
                    'internal_category_id_to_external_category_id_maps'
                ],
                'external_category_id_to_internal_category_id_maps': prompt_dict[
                    'external_category_id_to_internal_category_id_maps'
                ],
            }

        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor()
        return out

    def forward_get_prototypes(
        self,
        samples: t.Union[torch.Tensor, misc.NestedTensor],
        tilt_angles: torch.Tensor,
        group_start_indices: t.List[int],
        prompts: t.List[t.Union[torch.Tensor, t.List]],
        categories: t.List[t.Union[torch.Tensor, t.List]],
        tomogram_sizes: torch.Tensor,
    ) -> t.Dict[str, t.Union[torch.Tensor, t.Dict]]:
        group_start_indices = group_start_indices.copy()
        samples = self.check_input_and_process_samples(
            samples, tilt_angles, prompts, categories, tomogram_sizes,
        )
        self.set_image_tensor(samples)
        srcs, masks = self.forward_backbone(samples)
        encoder_outputs = self.transformer.forward_encoder(
            srcs=srcs,
            masks=masks,
            pos_embeds=self.poss,
            tilt_angles=tilt_angles,
            group_start_indices=group_start_indices,
            prompts=prompts,
            categories=categories,
        )
        self.unset_image_tensor()
        return {
            'group_prompt_dict': encoder_outputs['group_prompt_dict'],
        }

    def forward_with_prototypes(
        self,
        samples: t.Union[torch.Tensor, misc.NestedTensor],
        tilt_angles: torch.Tensor,
        group_start_indices: t.List[int],
        tomogram_sizes: torch.Tensor,
        prompt_embed_dict: t.Dict,
        **kw,
    ) -> t.Dict[str, t.Union[torch.Tensor, t.Dict]]:
        group_start_indices = group_start_indices.copy()
        samples = self.check_input_and_process_samples(
            samples, tilt_angles, None, None, tomogram_sizes,
        )
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        srcs, masks = self.forward_backbone(samples)

        encoder_outputs = self.transformer.forward_encoder(
            srcs=srcs,
            masks=masks,
            pos_embeds=self.poss,
            tilt_angles=tilt_angles,
            group_start_indices=group_start_indices,
            prompts=None,
            categories=None,
            prompt_embed_dict=prompt_embed_dict,
        )

        input_query_bbox = input_query_label = attn_mask = None

        _, group_prompt_dict, hs, reference, _, _, _ = self.transformer.forward_decoder(
            encoder_outputs=encoder_outputs,
            tilt_angles=tilt_angles,
            tomogram_sizes=tomogram_sizes,
            input_query_bbox=input_query_bbox,
            input_query_label=input_query_label,
            attn_mask=attn_mask,
        )

        outputs_coord_list = []
        for (layer_ref_sig, layer_bbox_embed, layer_hs) in zip(
            reference[:-1], self.bbox_embed3d, hs,
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + misc.inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, group_prompt_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        logits_group = outputs_class[-1]
        boxes_3d_group = outputs_coord_list[-1]
        out = {
            'pred_logits_group': logits_group,
            'pred_boxes_3d': boxes_3d_group,
            'group_internal_map': group_prompt_dict[
                'internal_category_id_to_external_category_id_maps'
            ],
            'group_external_map': group_prompt_dict[
                'external_category_id_to_internal_category_id_maps'
            ],
        }

        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor()
        return out


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha, focal_gamma, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative
                weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted
        non-empty boxes. This is not really a loss, it is intended for logging purposes only.
        It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image
        size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_labels(
        self,
        outputs: t.Dict[str, torch.Tensor],
        targets: t.List[t.Dict[str, t.Union[torch.Tensor, torch.Tensor]]],
        indices: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        log=True,
    ):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, 900, 256]
        num_classes = src_logits.shape[-1]
        prompt_token_mask = outputs.get('prompt_token_mask', None)

        # ([batch_ids], [query_ids]) e.g., ([0, 0, 0, 1], [5, 3, 1, 2])
        idx = self._get_src_permutation_idx(indices)

        # [number of bbox in batch]
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            num_classes,  # TODO: # of classes
            dtype=torch.int64,
            device=src_logits.device,
        )  # [B, 900]
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                src_logits.shape[0],
                src_logits.shape[1],
                src_logits.shape[2] + 1,
            ],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )  # [B, 900, 257]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        if prompt_token_mask is not None:
            prompt_token_mask = prompt_token_mask.repeat(
                1, src_logits.size(1),
            ).view(
                prompt_token_mask.shape[0], -1, prompt_token_mask.shape[1],
            )  # [B, 256] -> [B, 900, 256]
            src_logits = src_logits[prompt_token_mask]
            target_classes_onehot = target_classes_onehot[prompt_token_mask]

        loss_ce = utils.sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        )  # * src_logits.shape[1] # TODO: do we need this?
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes_3d(self, outputs, targets, indices, num_boxes):
        """
        Compute 3D L1 Loss
        outputs: remapped dict containing 'pred_boxes' (which are actually 3D boxes)
        targets: remapped list of dicts containing 'boxes' (which are actually 3D GT)
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]  # [K, 5]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0,
        )  # [K, 5]

        # Calculate L1 Loss on 5 dimensions (cx, cy, cz, w, h)
        loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox_3d'] = loss_l1.sum() / num_boxes

        return losses

    def loss_labels_3d(self, outputs, targets, indices, num_boxes):
        """
        Compute 3D Classification Loss (Reuse logic but return specific key)
        outputs: remapped dict containing 'pred_logits' (Group Level)
        targets: remapped list of dicts containing 'labels'
        """
        # Reuse the logic of loss_labels, but we want the key to be 'loss_ce_3d'
        # Call standard loss_labels (it will compute based on the inputs we give it)
        loss_dict = self.loss_labels(outputs, targets, indices, num_boxes, log=False)

        # Rename key for weight_dict compatibility
        return {'loss_ce_3d': loss_dict['loss_ce']}

    def loss_dn(self, outputs, num_boxes_3d):
        """
        Compute Contrastive Denoising Loss (Box L1 + Focal Class)
        outputs['dn_outputs'] must exist
        """
        if 'dn_outputs' not in outputs:
            return {}

        dn_out = outputs['dn_outputs']
        dn_meta = dn_out['dn_meta']

        valid_mask = dn_meta['valid_mask']  # [B, Pad_DN]
        positive_mask = dn_meta['positive_mask']  # [B, Pad_DN]

        prompt_token_mask = outputs['prompt_token_mask_group']

        # Valid AND Positive (Only these calculate Box Loss)
        valid_pos_mask = valid_mask & positive_mask

        # Meta Info
        mapped_targets = dn_meta['targets']  # Already mapped to Internal ID
        scalar = dn_meta['scalar']

        bs, pad_dn = valid_mask.shape
        device = valid_mask.device

        # 1. Prepare GT Tensors (Reconstruct GT for DN queries)
        # gt_labels_tensor: [B, Pad_DN] (Internal ID)
        # gt_boxes_tensor: [B, Pad_DN, 5]
        gt_labels_tensor = torch.zeros((bs, pad_dn), dtype=torch.long, device=device)
        gt_boxes_tensor = torch.zeros((bs, pad_dn, 5), dtype=torch.float32, device=device)

        for i, mapped_target in enumerate(mapped_targets):
            # t['labels']: [N_gt], t['boxes_3d']: [N_gt, 5]
            # Repeat scalar * 2 times (Pos + Neg)
            lbls = mapped_target['labels'].repeat(scalar * 2)
            bxs = mapped_target['boxes_3d'].repeat(scalar * 2, 1)

            num_k = len(lbls)
            if num_k > 0:
                if num_k > pad_dn:  # Truncate if needed
                    raise ValueError(f'num_k > pad_dn: {num_k} > {pad_dn}')
                    # lbls = lbls[:pad_dn]
                    # bxs = bxs[:pad_dn]
                    # num_k = pad_dn
                gt_labels_tensor[i, :num_k] = lbls
                gt_boxes_tensor[i, :num_k] = bxs

        # 2. Compute Loss for All Layers
        # dn_out['pred_logits']: [Layers, B, Pad_DN, C]
        dn_logits_all = dn_out['pred_logits']
        dn_boxes_all = dn_out['pred_boxes']
        num_layers = len(dn_logits_all)

        losses = {}

        # Precalculate Batch Indices for taking values from prompt_token_mask
        # valid_mask: [B, Pad_DN] -> nonzero -> [N_valid, 2] (batch_idx, dn_idx)
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)
        # [N_valid] corresponding Batch ID for every valid query
        batch_indices_flat = valid_indices[0]

        for l_id in range(num_layers):
            logits = dn_logits_all[l_id]
            boxes = dn_boxes_all[l_id]

            # Determine suffix: Last layer has no suffix, others are aux
            if l_id == num_layers - 1:
                suffix = ''
            else:
                suffix = f'_{l_id}'

            # --- A. Box Loss (Only Positive) ---
            if 'loss_bbox_3d' in self.weight_dict:
                src_boxes_pos = boxes[valid_pos_mask]
                tgt_boxes_pos = gt_boxes_tensor[valid_pos_mask]

                if src_boxes_pos.numel() > 0:
                    loss_bbox = F.l1_loss(src_boxes_pos, tgt_boxes_pos, reduction='none')
                    # Normalize by num_boxes_3d (same as matching loss)
                    losses[f'loss_bbox_3d_dn{suffix}'] = loss_bbox.sum() / num_boxes_3d
                else:
                    losses[f'loss_bbox_3d_dn{suffix}'] = torch.tensor(0.0, device=device)

            # --- B. Class Loss (Pos & Neg) ---
            if 'loss_ce_3d' in self.weight_dict:
                src_logits_valid = logits[valid_mask]  # [N_valid, C]

                # Default Target: All Zeros (Background)
                target_onehot = torch.zeros_like(src_logits_valid)

                # Fill Positive Targets
                # Get indices of positive samples relative to the flattened valid array
                pos_indices_in_valid = positive_mask[valid_mask]
                labels_valid = gt_labels_tensor[valid_mask]
                labels_pos = labels_valid[pos_indices_in_valid]

                if labels_pos.numel() > 0:
                    # Scatter 1.0 into correct class indices
                    target_onehot_pos = target_onehot[pos_indices_in_valid]
                    target_onehot_pos.scatter_(1, labels_pos.unsqueeze(-1), 1.0)
                    target_onehot[pos_indices_in_valid] = target_onehot_pos

                if prompt_token_mask is not None:
                    # According to Batch Index, take corresponding Mask
                    # prompt_token_mask: [B, 256]
                    # batch_indices_flat: [N_valid]
                    # mask_flat: [N_valid, 256]
                    mask_flat = prompt_token_mask[batch_indices_flat]

                    # Flatten & Filter
                    # [N_valid, 256] -> [N_total_valid_tokens]
                    # Only Mask=True preserved; -inf are discarded
                    src_logits_filtered = src_logits_valid[mask_flat]
                    target_onehot_filtered = target_onehot[mask_flat]
                else:
                    # Fallback (should not go here)
                    src_logits_filtered = src_logits_valid.flatten()
                    target_onehot_filtered = target_onehot.flatten()

                # Calculate Loss
                # sigmoid_focal_loss support any dim input
                if src_logits_filtered.numel() > 0:
                    loss_ce = utils.sigmoid_focal_loss(
                        src_logits_filtered,
                        target_onehot_filtered,
                        num_boxes_3d,
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                    )
                    losses[f'loss_ce_3d_dn{suffix}'] = loss_ce
                else:
                    losses[f'loss_ce_3d_dn{suffix}'] = torch.tensor(0.0, device=device)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'loss_bbox_3d': self.loss_boxes_3d,
            'loss_ce_3d': self.loss_labels_3d,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device

        # Important! First, map the category_id in targets to the internal one (for ordered prompts)
        # targets: external -> internal
        for target, mapping in zip(
            targets, outputs['external_category_id_to_internal_category_id_maps']
        ):
            labels = target['labels'].clone()
            new_labels = torch.tensor(
                [mapping[int(label.item())] for label in labels],
                device=labels.device,
                dtype=labels.dtype,
            )
            target['labels'] = new_labels

        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t['labels']) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if misc.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / misc.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            print(f'loss: {loss}')
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'gt_3d' in outputs:
            gt_3d_mapped = []
            group_maps = outputs['group_external_map']

            for gt_3d, mapping in zip(outputs['gt_3d'], group_maps):
                t_mapped = gt_3d.copy()
                raw_labels = gt_3d['labels']

                new_labels = torch.tensor(
                    [mapping[int(label.item())] for label in raw_labels],
                    device=raw_labels.device,
                    dtype=raw_labels.dtype,
                )
                t_mapped['labels'] = new_labels
                gt_3d_mapped.append(t_mapped)

            outputs_3d = {
                'pred_logits': outputs['pred_logits_group'],  # [B_groups, Nq, C]
                'pred_boxes': outputs['pred_boxes_3d'],  # [B_groups, Nq, 5]
                'prompt_token_mask': outputs.get('prompt_token_mask_group')  # Needed for Focal Loss
            }

            targets_3d = []
            for sub_gt_3d_mapped in gt_3d_mapped:
                targets_3d.append({
                    'boxes': sub_gt_3d_mapped['boxes_3d'],  # [N_gt, 5] -> Mapped to 'boxes'
                    'labels': sub_gt_3d_mapped['labels']
                })

            # C. 3D Matching (Once)
            indices_3d = self.matcher(outputs_3d, targets_3d)

            # D. 3D Num Boxes
            num_boxes_3d = sum(len(t['labels']) for t in targets_3d)
            num_boxes_3d = torch.as_tensor([num_boxes_3d], dtype=torch.float, device=device)
            if misc.is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_3d)
            num_boxes_3d = torch.clamp(num_boxes_3d / misc.get_world_size(), min=1).item()

            # E. Compute 3D Losses
            # Use get_loss with remapped inputs
            if 'loss_bbox_3d' in self.weight_dict:
                l_dict = self.get_loss(
                    'loss_bbox_3d', outputs_3d, targets_3d, indices_3d, num_boxes_3d
                )
                losses.update(l_dict)

            if 'loss_ce_3d' in self.weight_dict:  # Ensure this key is in build_groundingdino
                l_dict = self.get_loss(
                    'loss_ce_3d', outputs_3d, targets_3d, indices_3d, num_boxes_3d
                )
                losses.update(l_dict)

            if 'dn_outputs' in outputs:
                # Use same num_boxes_3d for normalization
                dn_losses = self.loss_dn(outputs, num_boxes_3d)
                losses.update(dn_losses)

        # In case of auxiliary losses, we repeat this process with the output of each
        # intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # Aux 3D Loss
                if 'gt_3d' in outputs:
                    # Remap Aux Outputs for 3D
                    aux_outputs_3d = {
                        'pred_logits': aux_outputs['pred_logits_group'],
                        'pred_boxes': aux_outputs['pred_boxes_3d'],
                        'prompt_token_mask': aux_outputs.get('prompt_token_mask_group')
                    }

                    # Match
                    indices_aux_3d = self.matcher(aux_outputs_3d, targets_3d)

                    # Compute Aux 3D Losses
                    if 'loss_bbox_3d' in self.weight_dict:
                        l_dict = self.get_loss(
                            'loss_bbox_3d', aux_outputs_3d, targets_3d, indices_aux_3d, num_boxes_3d
                        )
                        losses.update({k + f'_{idx}': v for k, v in l_dict.items()})

                    if 'loss_ce_3d' in self.weight_dict:
                        l_dict = self.get_loss(
                            'loss_ce_3d', aux_outputs_3d, targets_3d, indices_aux_3d, num_boxes_3d
                        )
                        losses.update({k + f'_{idx}': v for k, v in l_dict.items()})

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
            if 'gt_3d' in outputs:
                interm_outputs_3d = {
                    'pred_logits': interm_outputs['pred_logits_group'],
                    'pred_boxes': interm_outputs['pred_boxes_3d'],
                    'prompt_token_mask': interm_outputs.get('prompt_token_mask_group')
                }
                indices_interm_3d = self.matcher(interm_outputs_3d, targets_3d)

                if 'loss_bbox_3d' in self.weight_dict:
                    l_dict = self.get_loss(
                        'loss_bbox_3d',
                        interm_outputs_3d,
                        targets_3d,
                        indices_interm_3d,
                        num_boxes_3d,
                    )
                    losses.update({k + '_interm': v for k, v in l_dict.items()})

                if 'loss_ce_3d' in self.weight_dict:
                    l_dict = self.get_loss(
                        'loss_ce_3d',
                        interm_outputs_3d,
                        targets_3d,
                        indices_interm_3d,
                        num_boxes_3d
                    )
                    losses.update({k + '_interm': v for k, v in l_dict.items()})

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess3D(nn.Module):
    def __init__(
        self,
        num_select: int = 900,
        one_class_per_proposal: bool = False,
        nms_threshold: float = -1.0,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.one_class_per_proposal = one_class_per_proposal
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def perform_adaptive_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        nms_threshold: float,
    ) -> torch.Tensor:
        """
        Performs adaptive NMS on the estimated boxes based on their sizes.
        Args:
            boxes: Tensor of shape [N, 5] containing the predicted boxes in the format
                (x, y, z, w, h).
            scores: Tensor of shape [N] containing the confidence scores for each box.
            nms_threshold: Base distance threshold for NMS. The actual threshold for each pair of
                boxes will be (d_i + d_j) * nms_threshold, where d_i and d_j are the diameters of
                the boxes. A negative value means no NMS will be applied. Larger values will result
                in more aggressive suppression. For example, a value of 0.2 means that two boxes
                will be suppressed if the distance between their centers is less than 20% of the sum
                of their diameters.
        Returns:
            keep: Tensor containing the indices of the boxes to keep after NMS.
        """
        if boxes.shape[0] == 0 or nms_threshold < 0:
            return torch.arange(boxes.shape[0], device=boxes.device)

        # Calculate diameters: d = (h + w) / 2
        diameters = (boxes[:, 3] + boxes[:, 4]) / 2.0
        centers = boxes[:, :3]

        # Sort by confidence
        _, sorted_idx = scores.sort(0, descending=True)

        keep = []
        remaining = sorted_idx

        while remaining.numel() > 0:
            i = remaining[0]
            keep.append(i.item())

            if remaining.numel() == 1:
                break

            # Distance between current center and all others
            current_center = centers[i]
            other_centers = centers[remaining[1:]]
            dist = torch.norm(other_centers - current_center, dim=1)

            # Adaptive threshold: (d_i + d_j) * threshold
            current_diameter = diameters[i]
            other_diameters = diameters[remaining[1:]]
            adaptive_limit = (current_diameter + other_diameters) * nms_threshold

            # Mask out those that are too close
            mask = dist >= adaptive_limit
            remaining = remaining[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    @torch.no_grad()
    def forward(
        self,
        outputs: t.Dict[str, torch.Tensor],
        target_sizes: torch.Tensor,
        num_select: t.Optional[int] = None,
        nms_threshold: t.Optional[float] = None,
    ) -> t.List[t.Dict[str, torch.Tensor]]:
        """
        Converts the model's output into final 3D bounding box estimates.
        Args:
            outputs: A dictionary containing the model's raw outputs, including:
                - 'pred_logits_group': Tensor of shape [B, Q, C] with raw class scores for each
                    query.
                - 'pred_boxes_3d': Tensor of shape [B, Q, 5] with raw bounding box predictions
                    (x, y, z, w, h).
                - 'group_internal_map': List of dictionaries mapping internal category IDs to
                    external category IDs.
            target_sizes: Tensor of shape [B, 3] with the original tomogram sizes (W, H, D) for each
                sample in the batch.
            num_select: Optional integer specifying how many top estimates to select. If None, uses
                self.num_select.
            nms_threshold: Optional float specifying the NMS threshold to use. If None, uses
                self.nms_threshold.
        Returns:
            A list of dictionaries, one per batch item, each containing:
                - 'scores': Tensor of shape [K] with the confidence scores of the selected
                    predictions.
                - 'labels': Tensor of shape [K] with the external category IDs of the selected
                    predictions.
                - 'boxes_3d': Tensor of shape [K, 5] with the 3D bounding box coordinates of the
                    selected predictions, scaled to the original tomogram size.
        """

        out_logits = outputs['pred_logits_group']
        out_bbox = outputs['pred_boxes_3d']
        id_maps = outputs['group_internal_map']

        if len(out_logits) != len(target_sizes):
            raise ValueError('Output batch size and target_sizes batch size should be the same.')

        if target_sizes.shape[1] != 3:
            raise ValueError('target_sizes should have shape [B, 3] for (W, H, D).')

        if num_select is None:
            num_select = self.num_select

        if nms_threshold is None:
            nms_threshold = self.nms_threshold

        B, Q, C = out_logits.shape

        prob = out_logits.sigmoid()

        if self.one_class_per_proposal:
            max_scores, max_labels_idx = prob.max(dim=-1)
            k = min(num_select, Q)
            topk_values, topk_query_indexes = torch.topk(max_scores, k, dim=1)
            scores = topk_values
            topk_boxes_idx = topk_query_indexes
            labels_int = torch.gather(max_labels_idx, 1, topk_query_indexes)

        else:
            k = min(num_select, Q * C)
            topk_values, topk_indexes = torch.topk(
                prob.view(B, -1), k, dim=1
            )
            scores = topk_values
            topk_boxes_idx = torch.div(topk_indexes, C, rounding_mode='trunc')
            labels_int = topk_indexes % C

        boxes = torch.gather(
            out_bbox, 1, topk_boxes_idx.unsqueeze(-1).repeat(1, 1, 5)
        )
        img_w = target_sizes[:, 0]
        img_h = target_sizes[:, 1]
        img_d = target_sizes[:, 2]

        scale_fct = torch.stack([img_w, img_h, img_d, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        map_table = torch.full(
            (B, C), fill_value=-1, dtype=labels_int.dtype, device=labels_int.device
        )
        for b, d in enumerate(id_maps):
            for k, v in d.items():
                if 0 <= k < C:
                    map_table[b, k] = v

        labels_ext = torch.gather(map_table, 1, labels_int)

        results = []
        for b in range(B):
            if torch.any(labels_ext[b] < 0):
                invalid_ids = labels_int[b][labels_ext[b] < 0].tolist()
                raise ValueError(f'Invalid label mapping in batch {b}: {invalid_ids}')

            if nms_threshold >= 0:
                keep = self.perform_adaptive_nms(boxes[b], scores[b], nms_threshold)
                b_scores = scores[b][keep]
                b_labels = labels_ext[b][keep]
                b_boxes = boxes[b][keep]

            else:
                b_scores = scores[b]
                b_labels = labels_ext[b]
                b_boxes = boxes[b]

            results.append({
                'scores': b_scores,
                'labels': b_labels,
                'boxes_3d': b_boxes,
            })

        return results

    @torch.no_grad()
    def scale_back_gt(
        self,
        gt_batch: t.List[t.Dict[str, torch.Tensor]],
    ) -> t.List[t.Dict[str, torch.Tensor]]:
        return eval_3d.scale_back_gt(gt_batch)


@registry.MODULE_BUILD_FUNCS.registe_with_name(module_name='fulltilt')
def build_fulltilt(args, training=False):
    device = torch.device(args.device)
    backbone = fulltilt_backbone.build_backbone(args)
    transformer = fulltilt_transformer.build_transformer(args)

    model = FullTilt(
        backbone,
        transformer,
        num_queries=args.num_queries,
        query_dim=5,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        max_unique_classes=args.max_unique_classes,
        aux_loss=args.aux_loss,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
    )
    if not training:
        return model, None, None

    matcher = fulltilt_matcher.build_matcher(args)
    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # Add 3D loss weight (default to bbox_loss_coef if not specified)
    bbox_3d_coef = getattr(args, 'bbox_3d_loss_coef', None)
    assert bbox_3d_coef is not None
    weight_dict['loss_bbox_3d'] = bbox_3d_coef

    ce_3d_coef = getattr(args, 'ce_3d_loss_coef', None)
    assert ce_3d_coef is not None
    weight_dict['loss_ce_3d'] = ce_3d_coef

    # If args.dn_number > 0 set weights for loss_bbox_3d_dn
    if args.dn_number > 0:
        weight_dict['loss_bbox_3d_dn'] = bbox_3d_coef
        weight_dict['loss_ce_3d_dn'] = ce_3d_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    interm_weight_dict = {}
    try:
        no_interm_box_loss = args.no_interm_box_loss
    except Exception:
        no_interm_box_loss = False
    _coeff_weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
        'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        'loss_bbox_3d': 1.0 if not no_interm_box_loss else 0.0,
        'loss_ce_3d': 1.0 if not no_interm_box_loss else 0.0,
    }
    try:
        interm_loss_coef = args.interm_loss_coef
    except Exception:
        interm_loss_coef = 1.0
    for k, v in clean_weight_dict.items():
        if 'dn' in k:
            continue  # Skip DN keys for interm
        if k in _coeff_weight_dict:
            interm_weight_dict[k + '_interm'] = v * interm_loss_coef * _coeff_weight_dict[k]
    weight_dict.update(interm_weight_dict)

    losses = []  # NOTE: 3D losses are hardcoded

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        losses=losses,
    )
    criterion.to(device)
    processor_3d = PostProcess3D(num_select=args.num_select)

    return model, criterion, processor_3d
