# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import typing as t

import fulltilt.datasets.coco as coco
import fulltilt.datasets.random_coco as random_coco


def build_dataset(image_set: str, args: argparse.Namespace, datasetinfo: t.Dict[str, str]):
    if datasetinfo['dataset_mode'] == 'coco':
        return coco.build(image_set, args, datasetinfo)

    if datasetinfo['dataset_mode'] == 'random_coco':
        return random_coco.build_coco_random_training(image_set, args, datasetinfo)

    raise ValueError(f'Dataset {args.dataset_file} not supported')
