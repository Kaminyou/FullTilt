# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None,
    distributed_rank=0,
    *,
    color=False,
    name="imagenet",
    abbrev_name=None,
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    if color:
        raise NotImplementedError('not support color=True')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d]: %(message)s', datefmt='%m/%d %H:%M:%S',
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')
        if distributed_rank > 0:
            filename = filename + f'.rank{distributed_rank}'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, 'a')
