# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

import typing as t

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: t.Union[float, torch.Tensor] = 1e-5, inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
