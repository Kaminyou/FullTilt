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
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import typing as t

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from fulltilt.models.FullTilt import (block, ms_deform_attn, tilt_angle_embed,
                                      transformer_vanilla, utils)
from fulltilt.util import misc


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = 'relu',
        query_dim: int = 5,
        num_feature_levels: int = 1,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        max_unique_classes: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_unique_classes = max_unique_classes

        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.tilt_encoder = tilt_angle_embed.TiltEncoder(d_model=d_model)
        self.frame_blocks = nn.ModuleList(
            [
                block.Block(
                    dim=256,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=None,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block.Block(
                    dim=256,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=None,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.category_embed = nn.Parameter(torch.Tensor(2, d_model))
        self.category_embed_proj = nn.Linear(768, 256)
        self.category_pos_proj = nn.Linear(512, 256)

        prompt_encoder = DeformableTransformerPromptEncoderLayer(
            d_model,
            dim_feedforward,
            dropout, activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.prompt_encoders = utils._get_clones(
            prompt_encoder,
            num_encoder_layers,
            layer_share=False,
        )

        prompt_self_attn_layer = transformer_vanilla.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
        )
        self.prompt_self_attn_layers = utils._get_clones(
            prompt_self_attn_layer,
            num_encoder_layers,
            layer_share=False,
        )

        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)

        self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder3D(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, ms_deform_attn.MultiScaleDeformableAttention):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask: torch.Tensor) -> torch.Tensor:
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _process_frame_attention(
        self, tokens: torch.Tensor,
        frame_idx: int,
        pos: t.Optional[torch.Tensor] = None,
    ) -> t.Tuple[torch.Tensor, int]:
        if self.training:
            tokens = checkpoint.checkpoint(
                self.frame_blocks[frame_idx],
                tokens,
                pos,
                use_reentrant=False,
            )
        else:
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
        frame_idx += 1

        return tokens, frame_idx

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        global_idx: int,
        group_start_end_indices: t.List[t.Tuple[int, int]],
        spatial_shapes: torch.Tensor,
        pos: t.Optional[torch.Tensor] = None,
    ) -> t.Tuple[torch.Tensor, int]:
        tokens_collection = []

        split_sections = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()

        for group_start_end_index in group_start_end_indices:
            start, end = group_start_end_index

            sub_tokens = tokens[start:end]
            _B, _, _C = sub_tokens.shape

            if pos is not None:
                sub_pos = pos[start:end]
            else:
                sub_pos = None

            level_tokens_list = torch.split(sub_tokens, split_sections, dim=1)

            if sub_pos is not None:
                level_pos_list = torch.split(sub_pos, split_sections, dim=1)
            else:
                level_pos_list = [None] * len(level_tokens_list)

            processed_levels = []

            for i, (lvl_tokens, lvl_pos) in enumerate(zip(level_tokens_list, level_pos_list)):
                H, W = int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1])

                lvl_tokens = lvl_tokens.view(_B, H, W, _C)
                if lvl_pos is not None:
                    lvl_pos = lvl_pos.view(_B, H, W, _C)

                lvl_tokens = lvl_tokens.permute(1, 0, 2, 3)
                if lvl_pos is not None:
                    lvl_pos = lvl_pos.permute(1, 0, 2, 3)

                lvl_tokens = lvl_tokens.flatten(1, 2)
                if lvl_pos is not None:
                    lvl_pos = lvl_pos.flatten(1, 2)

                if self.training:
                    lvl_tokens = checkpoint.checkpoint(
                        self.global_blocks[global_idx],
                        lvl_tokens,
                        lvl_pos,
                        use_reentrant=False,
                    )
                else:
                    lvl_tokens = self.global_blocks[global_idx](lvl_tokens, pos=lvl_pos)

                lvl_tokens = lvl_tokens.view(H, _B, W, _C)
                lvl_tokens = lvl_tokens.permute(1, 0, 2, 3)
                lvl_tokens = lvl_tokens.flatten(1, 2)

                processed_levels.append(lvl_tokens)

            group_output = torch.cat(processed_levels, dim=1)
            tokens_collection.append(group_output)

        global_idx += 1
        return torch.cat(tokens_collection), global_idx

    def get_visual_prompt_embedding_optimized(
        self,
        prompts: t.List[t.Union[torch.Tensor, t.List]],
        categories: t.List[t.Union[torch.Tensor, t.List]],
        src_flatten: torch.Tensor,
        mask_flatten: torch.Tensor,
        group_start_end_indices: t.List[t.Tuple[int, int]],
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> t.Tuple[t.Dict[str, torch.Tensor], t.Dict[str, t.Union[torch.Tensor, t.List[t.Dict]]]]:
        internal_category_id_to_external_category_id_maps = []
        external_category_id_to_internal_category_id_maps = []
        encoded_categories = []
        categories_token_mask = []
        categories_self_attn_mask = []

        group_internal_category_id_to_external_category_id_maps = []
        group_external_category_id_to_internal_category_id_maps = []
        group_encoded_categories = []
        group_categories_token_mask = []
        group_categories_self_attn_mask = []

        _default_pos = torch.tensor([0.5, 0.5, 1.0, 1.0], device=src_flatten.device)[None, None]
        _default_pos_embed = utils.gen_sineembed_for_position(_default_pos).squeeze(0)

        type_embed_prompt = self.category_embed[1][None, None]

        for group_start_end_index in group_start_end_indices:
            start, end = group_start_end_index
            n_imgs = end - start

            group_cats = set()
            for i in range(start, end):
                c_data = categories[i]
                c_list = c_data.tolist() if torch.is_tensor(c_data) else c_data
                for c in c_list:
                    group_cats.add(int(c))

            sorted_cats = sorted(list(group_cats))
            num_unique_cats = len(sorted_cats)
            ext_to_int = {c: i for i, c in enumerate(sorted_cats)}
            int_to_ext = {i: c for i, c in enumerate(sorted_cats)}

            sub_prompts = prompts[start:end]
            sub_cats = categories[start:end]
            num_prompts_per_img = [len(p) for p in sub_prompts]
            max_prompts = max(num_prompts_per_img) if num_prompts_per_img else 0

            seq_len = max_prompts + num_unique_cats

            batch_prompts = torch.zeros((n_imgs, max_prompts, 4), device=src_flatten.device)
            batch_cat_indices = torch.full(
                (n_imgs, max_prompts), -1, dtype=torch.long, device=src_flatten.device,
            )

            batch_key_padding_mask = torch.ones(
                (n_imgs, seq_len), dtype=torch.bool, device=src_flatten.device,
            )
            batch_cat_presence = torch.zeros(
                (n_imgs, num_unique_cats), dtype=torch.bool, device=src_flatten.device,
            )

            for i in range(n_imgs):
                p_list = sub_prompts[i]
                c_list = sub_cats[i]
                c_list = c_list.tolist() if torch.is_tensor(c_list) else c_list
                k = len(p_list)

                if k > 0:
                    batch_prompts[i, :k] = torch.as_tensor(p_list, device=src_flatten.device)
                    batch_key_padding_mask[i, :k] = False

                    local_indices = [ext_to_int[int(c)] for c in c_list]
                    batch_cat_indices[i, :k] = torch.tensor(
                        local_indices, dtype=torch.long, device=src_flatten.device,
                    )

                    for loc_idx in local_indices:
                        batch_cat_presence[i, loc_idx] = True

            batch_key_padding_mask[:, max_prompts:] = False

            cat_id_seq = torch.full(
                (n_imgs, seq_len), -1, dtype=torch.long, device=src_flatten.device
            )
            cat_id_seq[:, :max_prompts] = batch_cat_indices

            proto_ids = torch.arange(
                num_unique_cats, device=src_flatten.device,
            ).unsqueeze(0).expand(n_imgs, -1)
            cat_id_seq[:, max_prompts:] = proto_ids

            adjacency = (cat_id_seq.unsqueeze(-1) == cat_id_seq.unsqueeze(1))
            is_valid = (cat_id_seq != -1).unsqueeze(-1) & (cat_id_seq != -1).unsqueeze(1)
            adjacency = adjacency & is_valid

            attn_mask = torch.zeros((n_imgs, seq_len, seq_len), device=src_flatten.device)
            attn_mask.masked_fill_(~adjacency, float('-inf'))
            attn_mask = attn_mask.repeat_interleave(self.nhead // 2, dim=0)

            batch_prompt_pos = utils.gen_sineembed_for_position(batch_prompts)

            current_prototypes = self.category_embed[0].unsqueeze(0).repeat(num_unique_cats, 1)

            sub_src = src_flatten[start:end]
            sub_mask = mask_flatten[start:end]

            for layer_idx in range(self.num_encoder_layers):
                content_prompts = type_embed_prompt.expand(n_imgs, max_prompts, -1)
                content_protos = current_prototypes.unsqueeze(0).expand(n_imgs, -1, -1)
                query_content = torch.cat([content_prompts, content_protos], dim=1)

                pos_prompts = batch_prompt_pos
                pos_protos = _default_pos_embed.unsqueeze(0).expand(n_imgs, num_unique_cats, -1)
                query_pos = torch.cat([pos_prompts, pos_protos], dim=1)

                query_embed = self.category_embed_proj(
                    torch.cat([query_content, query_pos], dim=-1),
                )

                ref_prompts = batch_prompts.unsqueeze(2)
                ref_protos = _default_pos.expand(n_imgs, num_unique_cats, -1, -1)
                ref_points = torch.cat([ref_prompts, ref_protos], dim=1)

                if self.training:
                    output = checkpoint.checkpoint(
                        self.prompt_encoders[layer_idx],
                        query_embed,
                        None,
                        sub_src,
                        ref_points,
                        spatial_shapes,
                        level_start_index,
                        sub_mask,
                        use_reentrant=False,
                    )
                else:
                    output = self.prompt_encoders[layer_idx](
                        src=query_embed,
                        pos=None,
                        value=sub_src,
                        reference_points=ref_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=sub_mask,
                    )

                pos_embed_short = self.category_pos_proj(query_pos)
                output = self.prompt_self_attn_layers[layer_idx](
                    src=output,
                    pos=pos_embed_short,
                    src_key_padding_mask=batch_key_padding_mask,
                    src_mask=attn_mask,
                )

                out_protos = output[:, max_prompts:, :]

                valid_out_protos = out_protos * batch_cat_presence.unsqueeze(-1)

                sum_protos = valid_out_protos.sum(dim=0)
                count_protos = batch_cat_presence.sum(dim=0, keepdim=True).T
                count_protos = count_protos.clamp(min=1.0)

                new_prototypes = sum_protos / count_protos
                current_prototypes = new_prototypes

            final_encoded = torch.zeros(
                (n_imgs, self.max_unique_classes, self.d_model), device=src_flatten.device,
            )
            final_mask = torch.zeros(
                (n_imgs, self.max_unique_classes), dtype=torch.bool, device=src_flatten.device,
            )
            final_attn_mask = torch.zeros(
                (n_imgs, self.max_unique_classes, self.max_unique_classes),
                dtype=torch.bool,
                device=src_flatten.device,
            )

            final_encoded[:, :num_unique_cats, :] = current_prototypes.unsqueeze(0)
            final_mask[:, :num_unique_cats] = True
            final_attn_mask[:, :num_unique_cats, :num_unique_cats] = True

            encoded_categories.append(final_encoded)
            categories_token_mask.append(final_mask)
            categories_self_attn_mask.append(final_attn_mask)

            internal_category_id_to_external_category_id_maps += [
                copy.deepcopy(int_to_ext) for _ in range(n_imgs)
            ]
            external_category_id_to_internal_category_id_maps += [
                copy.deepcopy(ext_to_int) for _ in range(n_imgs)
            ]

            group_encoded = torch.zeros(
                (1, self.max_unique_classes, self.d_model), device=src_flatten.device,
            )
            group_mask = torch.zeros(
                (1, self.max_unique_classes), dtype=torch.bool, device=src_flatten.device,
            )
            group_attn = torch.zeros(
                (1, self.max_unique_classes, self.max_unique_classes),
                dtype=torch.bool,
                device=src_flatten.device,
            )

            group_encoded[0, :num_unique_cats, :] = current_prototypes
            group_mask[0, :num_unique_cats] = True
            group_attn[0, :num_unique_cats, :num_unique_cats] = True

            group_encoded_categories.append(group_encoded)
            group_categories_token_mask.append(group_mask)
            group_categories_self_attn_mask.append(group_attn)

            group_internal_category_id_to_external_category_id_maps.append(
                copy.deepcopy(int_to_ext),
            )
            group_external_category_id_to_internal_category_id_maps.append(
                copy.deepcopy(ext_to_int),
            )

        encoded_categories = torch.cat(encoded_categories)
        categories_token_mask = torch.cat(categories_token_mask)
        categories_self_attn_mask = torch.cat(categories_self_attn_mask)

        group_encoded_categories = torch.cat(group_encoded_categories)
        group_categories_token_mask = torch.cat(group_categories_token_mask)
        group_categories_self_attn_mask = torch.cat(group_categories_self_attn_mask)

        prompt_dict = {
            'encoded_prompt': encoded_categories,
            'prompt_token_mask': categories_token_mask,
            'prompt_self_attention_masks': categories_self_attn_mask,
            'internal_category_id_to_external_category_id_maps': internal_category_id_to_external_category_id_maps,  # noqa
            'external_category_id_to_internal_category_id_maps': external_category_id_to_internal_category_id_maps,  # noqa
        }

        group_prompt_dict = {
            'encoded_prompt': group_encoded_categories,
            'prompt_token_mask': group_categories_token_mask,
            'prompt_self_attention_masks': group_categories_self_attn_mask,
            'internal_category_id_to_external_category_id_maps':
                group_internal_category_id_to_external_category_id_maps,
            'external_category_id_to_internal_category_id_maps':
                group_external_category_id_to_internal_category_id_maps,
        }

        return prompt_dict, group_prompt_dict

    def forward_encoder(
        self,
        srcs: t.List[torch.Tensor],
        masks: t.List[torch.Tensor],
        pos_embeds: t.List[torch.Tensor],
        tilt_angles: torch.Tensor,
        group_start_indices: t.List[int],
        prompts: t.Optional[t.List[t.Union[torch.Tensor, t.List]]] = None,
        categories: t.Optional[t.List[t.Union[torch.Tensor, t.List]]] = None,
        prompt_embed_dict: t.Optional[t.Dict] = None,
    ) -> t.Dict[str, t.Any]:
        if prompt_embed_dict is None:
            if prompts is None or categories is None:
                raise ValueError(
                    'If prompt_embed_dict is not provided, prompts and categories must be provided.'
                )

        if prompt_embed_dict is not None:
            if prompts is not None or categories is not None:
                raise ValueError(
                    'If prompt_embed_dict is provided, prompts and categories should not be'
                    'provided.'
                )

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        tilt_embeds = self.tilt_encoder(tilt_angles)
        tilt_embeds = tilt_embeds.unsqueeze(1)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten + tilt_embeds

        group_start_end_indices = []
        group_start_indices.append(len(tilt_angles))
        for i in range(len(group_start_indices) - 1):
            group_start_end_indices.append([group_start_indices[i], group_start_indices[i + 1]])

        frame_idx = 0
        global_idx = 0
        for _ in range(self.num_encoder_layers):
            src_flatten, frame_idx = self._process_frame_attention(
                src_flatten, frame_idx, pos=lvl_pos_embed_flatten,
            )
            src_flatten, global_idx = self._process_global_attention(
                src_flatten,
                global_idx,
                group_start_end_indices=group_start_end_indices,
                spatial_shapes=spatial_shapes,
                pos=lvl_pos_embed_flatten,
            )

        if prompt_embed_dict is not None:
            if len(group_start_end_indices) != 1:
                raise ValueError('Only supports 1 group for cross image prompting.')
            if len(prompt_embed_dict['group_prompt_dict']['encoded_prompt']) != 1:
                raise ValueError('Only supports prompt embeddings with dim0 = 1.')

            n_imgs = group_start_end_indices[0][1] - group_start_end_indices[0][0]
            group_prompt_dict = prompt_embed_dict['group_prompt_dict']
            prompt_dict = copy.deepcopy(group_prompt_dict)
            prompt_dict['encoded_prompt'] = prompt_dict['encoded_prompt'].repeat_interleave(
                n_imgs, dim=0,
            )
            prompt_dict['prompt_token_mask'] = prompt_dict['prompt_token_mask'].repeat_interleave(
                n_imgs, dim=0,
            )
            prompt_dict['prompt_self_attention_masks'] = prompt_dict[
                'prompt_self_attention_masks'
            ].repeat_interleave(n_imgs, dim=0)
            prompt_dict['internal_category_id_to_external_category_id_maps'] = prompt_dict[
                'internal_category_id_to_external_category_id_maps'
            ] * n_imgs
            prompt_dict['external_category_id_to_internal_category_id_maps'] = prompt_dict[
                'external_category_id_to_internal_category_id_maps'
            ] * n_imgs

        else:
            prompt_dict, group_prompt_dict = self.get_visual_prompt_embedding_optimized(
                prompts=prompts,
                categories=categories,
                src_flatten=src_flatten,
                mask_flatten=mask_flatten,
                group_start_end_indices=group_start_end_indices,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        return {
            'src_flatten': src_flatten,
            'mask_flatten': mask_flatten,
            'lvl_pos_embed_flatten': lvl_pos_embed_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
            'group_start_end_indices': group_start_end_indices,
            'prompt_dict': prompt_dict,
            'group_prompt_dict': group_prompt_dict
        }

    def forward_decoder(
        self,
        encoder_outputs: t.Dict[str, t.Any],
        tilt_angles: torch.Tensor,
        tomogram_sizes: torch.Tensor,
        input_query_bbox: t.Optional[torch.Tensor] = None,  # not used in inference
        input_query_label: t.Optional[torch.Tensor] = None,  # not used in inference
        attn_mask: t.Optional[torch.Tensor] = None,  # not used in inference
    ):
        src_flatten = encoder_outputs['src_flatten']
        mask_flatten = encoder_outputs['mask_flatten']
        spatial_shapes = encoder_outputs['spatial_shapes']
        level_start_index = encoder_outputs['level_start_index']
        valid_ratios = encoder_outputs['valid_ratios']
        group_start_end_indices = encoder_outputs['group_start_end_indices']
        prompt_dict = encoder_outputs['prompt_dict']
        group_prompt_dict = encoder_outputs['group_prompt_dict']

        bs_groups = len(group_start_end_indices)
        _tilts = tilt_angles.tolist() if torch.is_tensor(tilt_angles) else tilt_angles
        best_indices = []

        for start, end in group_start_end_indices:
            g_tilts = _tilts[start:end]
            local_min = min(range(len(g_tilts)), key=lambda x: abs(g_tilts[x]))
            best_indices.append(start + local_min)

        best_indices = torch.tensor(best_indices, device=src_flatten.device)

        output_memory, output_proposals = utils.gen_encoder_output_proposals(
            src_flatten, mask_flatten, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_coord_unselected = (
            self.enc_out_bbox_embed(output_memory) + output_proposals
        )

        group_memory = output_memory[best_indices]
        group_proposals_refined = enc_outputs_coord_unselected[best_indices]
        enc_outputs_class_unselected = self.enc_out_class_embed(group_memory, group_prompt_dict)

        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        invalid_proposals_mask = torch.logical_or(
            torch.isnan(group_proposals_refined).any(dim=-1),
            torch.isinf(group_proposals_refined).any(dim=-1)
        )

        if invalid_proposals_mask.any():
            topk_logits = topk_logits.masked_fill(invalid_proposals_mask, float('-inf'))
        topk = self.num_queries
        topk_proposals_idx = torch.topk(topk_logits, topk, dim=1)[1]
        ref_2d_raw = torch.gather(
            group_proposals_refined, 1, topk_proposals_idx.unsqueeze(-1).repeat(1, 1, 4)
        )
        z_fixed = torch.zeros_like(ref_2d_raw[..., :1])
        ref_3d_for_loss = torch.cat([
            ref_2d_raw[..., 0:2],
            z_fixed,
            ref_2d_raw[..., 2:4]
        ], dim=-1)

        ref_enc = ref_3d_for_loss.sigmoid().unsqueeze(0)

        ref_3d_for_decoder = torch.cat([
            ref_2d_raw[..., 0:2].detach(),
            z_fixed,
            ref_2d_raw[..., 2:4].detach(),
        ], dim=-1)

        refpoint_embed_ = ref_3d_for_decoder
        init_box_proposal = refpoint_embed_.sigmoid()

        tgt_ = self.tgt_embed.weight[None, :, :].repeat(bs_groups, 1, 1)

        refpoint_embed, tgt = refpoint_embed_, tgt_

        hs_enc_undetach = torch.gather(
            group_memory, 1, topk_proposals_idx.unsqueeze(-1).repeat(1, 1, self.d_model)
        )
        hs_enc = hs_enc_undetach.unsqueeze(0)

        if input_query_bbox is not None and input_query_label is not None:
            # Concat Content Query: [DN + Nq, B, C]
            tgt = torch.cat([input_query_label, tgt], dim=1)

            # Concat Reference Points: [B, DN+Nq, 5]
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)

        hs, references = self.decoder(
            tgt=tgt,  # [B_group, num_queries, C]
            memory=src_flatten,  # [B_total, h*w, C]
            memory_key_padding_mask=mask_flatten,  # [B_total, h*w]
            refpoints_unsigmoid=refpoint_embed,  # [B_group, num_queries, 4]
            spatial_shapes=spatial_shapes,  # [L, 2]
            level_start_index=level_start_index,  # [L]
            valid_ratios=valid_ratios,  # [B_group, L, 2]
            memory_prompt=prompt_dict['encoded_prompt'],  # [B_group, num_prompt_tokens, C]
            prompt_attention_mask=~prompt_dict['prompt_token_mask'],  # [B_group, num_prompt_tokens]
            tilt_angles=tilt_angles,
            group_start_end_indices=group_start_end_indices,
            tomogram_sizes=tomogram_sizes,
            tgt_mask=attn_mask,  # not used in inference
        )

        return prompt_dict, group_prompt_dict, hs, references, hs_enc, ref_enc, init_box_proposal


class DeformableTransformerPromptEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = ms_deform_attn.MultiScaleDeformableAttention(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = utils._get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src: torch.Tensor,
        pos: t.Optional[torch.Tensor],  # not used in fulltilt, but kept for potential future use
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: torch.Tensor,  # mask out value tokens
    ) -> torch.Tensor:
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        self.cross_attn = ms_deform_attn.MultiScaleDeformableAttention(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        self.ca_prompt = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ca_prompt_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ca_prompt_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = utils._get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        tgt_reference_points: torch.Tensor,
        memory_spatial_shapes: torch.Tensor,
        memory_level_start_index: torch.Tensor,
        memory_prompt: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        self_attn_mask: t.Optional[torch.Tensor] = None,  # not used in inference
    ):
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ca_prompt(
            self.with_pos_embed(tgt, tgt_query_pos),
            memory_prompt,
            memory_prompt,
            key_padding_mask=prompt_attention_mask,
        )[0]
        tgt = tgt + self.ca_prompt_dropout(tgt2)
        tgt = self.ca_prompt_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            reference_points=tgt_reference_points,
            value=memory,
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder3D(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module,
        d_model: int = 256,
        query_dim: int = 5,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError('num_layers should > 0')
        self.layers = utils._get_clones(decoder_layer, num_layers)
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = norm
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels

        sine_embed_dim = 128 * 5  # = 640
        self.ref_point_head3d = utils.MLP(sine_embed_dim, d_model, d_model, 2)

    def forward(
        self,
        tgt: torch.Tensor,  # Q # [B_group, Q, C]
        memory: torch.Tensor,  # V # [B_total, HW, C]
        memory_key_padding_mask: torch.Tensor,  # mask for V
        refpoints_unsigmoid: torch.Tensor,  # K # [B_group, Q, 5]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        memory_prompt: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        tilt_angles: torch.Tensor,
        group_start_end_indices: t.List[int],
        tomogram_sizes: torch.Tensor,
        tgt_mask: t.Optional[torch.Tensor] = None,  # not used in inference
    ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        device = output.device
        B_total = memory.shape[0]

        group_ids_per_img = torch.zeros(B_total, dtype=torch.long, device=device)
        group_counts = []
        for idx, group_start_end_index in enumerate(group_start_end_indices):
            start, end = group_start_end_index
            n_imgs = end - start
            group_counts.append(n_imgs)
            group_ids_per_img[start:end] = idx

        group_counts = torch.tensor(group_counts, device=tilt_angles.device)

        for layer_id, layer in enumerate(self.layers):
            output_expanded = output[group_ids_per_img]  # [B_total, Q, C]

            reference_points_expanded = reference_points[group_ids_per_img]  # [B_total, Q, 5]
            tomogram_sizes_expanded = tomogram_sizes.repeat_interleave(group_counts, dim=0)
            ref_2d_box = utils.project_3d_to_2d_batch(
                reference_points_expanded,
                tilt_angles,
                tomogram_sizes_expanded,
            )

            ratio_scale = torch.cat([valid_ratios, valid_ratios], dim=-1)

            reference_points_input = (
                ref_2d_box.unsqueeze(2) * ratio_scale.unsqueeze(1)
            )  # [B_total, Q, num_levels, 4]

            query_sine_embed = utils.gen_sineembed_for_position_3d(reference_points)
            raw_query_pos = self.ref_point_head3d(query_sine_embed)
            query_pos_expanded = raw_query_pos[group_ids_per_img]

            if self.training:
                output = checkpoint.checkpoint(
                    layer,
                    output_expanded,
                    query_pos_expanded,
                    memory,
                    memory_key_padding_mask,
                    reference_points_input,
                    spatial_shapes,
                    level_start_index,
                    memory_prompt,
                    prompt_attention_mask,
                    tgt_mask,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    tgt=output_expanded,
                    tgt_query_pos=query_pos_expanded,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_reference_points=reference_points_input,
                    memory_spatial_shapes=spatial_shapes,
                    memory_level_start_index=level_start_index,
                    memory_prompt=memory_prompt,
                    prompt_attention_mask=prompt_attention_mask,
                    self_attn_mask=tgt_mask,
                )

            if output.isnan().any() | output.isinf().any():
                print(f'output layer_id {layer_id} is nan')
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f'num_nan {num_nan}, num_inf {num_inf}')
                except Exception as e:
                    print(e)

            outputs_split = torch.split(output, group_counts, dim=0)
            agg_outputs = []
            for g_out in outputs_split:
                agg_outputs.append(g_out.mean(dim=0))

            output = torch.stack(agg_outputs, dim=0)
            delta_unsig = self.bbox_embed3d[layer_id](output)

            ref_unsig = misc.inverse_sigmoid(reference_points)
            ref_unsig = ref_unsig + delta_unsig
            new_reference_points = ref_unsig.sigmoid()

            reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return intermediate, ref_points


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        query_dim=5,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        max_unique_classes=args.max_unique_classes,
    )
