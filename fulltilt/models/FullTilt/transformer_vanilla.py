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
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0.
# All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import typing as t

import torch
import torch.nn as nn

from fulltilt.models.FullTilt import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = utils._get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: t.Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: t.Optional[torch.Tensor] = None,
        src_key_padding_mask: t.Optional[torch.Tensor] = None,
        pos: t.Optional[torch.Tensor] = None,
    ):
        """
        src: [Batch, Seq_Len, Dim] (batch_first=True)
        src_mask: [Batch*Heads, Seq_Len, Seq_Len] (Block Diagonal Mask)
        src_key_padding_mask: [Batch, Seq_Len] (Padding Mask)
        pos: [Batch, Seq_Len, Dim] (Positional Encoding)
        """

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(
            query=q,
            key=k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
