# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
#
# Note: This file contains modifications to the original works cited below.
# ========================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import collections
import datetime
import functools
import io
import json
import os
import pickle
import subprocess
import time
import typing as t

import torch
import torch.distributed as distributed
import torch.nn as nn
import torchvision


class NestedTensor(object):
    def __init__(self, tensors, mask: t.Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == 'auto':
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    'tensors dim must be 3 or 4 but {}({})'.format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, 'dim of tensor should be 3 but {}'.format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {'tensors.shape': self.tensors.shape, 'mask.shape': self.mask.shape}


def nested_tensor_from_tensor_list(tensor_list: t.List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: t.List[torch.Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def clean_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return distributed.get_rank()


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_dist_avail_and_initialized():
    if not distributed.is_available():
        return False
    if not distributed.is_initialized():
        return False
    return True


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def fix_num_query_weight(
    model: nn.Module,
    checkpoint: t.Dict[str, torch.Tensor],
):
    """
    Fixes the weight of the query embedding in the checkpoint to match the model's
    expected number of queries.
    """
    model_dict = model.state_dict()
    ckpt_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    keys_to_fix = ['transformer.tgt_embed.weight']
    for key in keys_to_fix:
        if key in ckpt_dict and key in model_dict:
            src_weight = ckpt_dict[key]
            target_size = model_dict[key].shape[0]
            src_size = src_weight.shape[0]

            if src_size != target_size:
                if src_size > target_size:
                    # If the checkpoint has more queries than the model, we can simply slice
                    new_weight = src_weight[:target_size]
                    ckpt_dict[key] = new_weight
                    print(f'Sliced {key}: {src_size} -> {target_size}')
                else:
                    # 1. Calculate how many full copies we need to cover the target
                    # Use math.ceil or (target // src) + 1
                    n_repeats = (target_size + src_size - 1) // src_size

                    # 2. Repeat and then slice to the exact target size
                    # .repeat(n_repeats, 1) handles the tiled copying
                    # [:target_size] chops off the extra if it wasn't a clean division
                    new_weight = src_weight.repeat(n_repeats, 1)[:target_size]

                    ckpt_dict[key] = new_weight
                    print(
                        f'Resized {key}: {src_size} -> {target_size}'
                        f' (using {n_repeats} repeats + slice)'
                    )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_distributed(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '':
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = args.rank * local_world_size + args.local_rank
        print(f'world size: {args.world_size}, rank: {args.rank}, local rank: {args.local_rank}')
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '23233'
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        print(
            f'world size: {args.world_size}, world rank: {args.rank}, '
            f'local rank: {args.local_rank}, device_count: {torch.cuda.device_count()}'
        )
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    print(f'  == distributed init (rank {args.rank}) done.')

    setup_for_distributed(args.rank == 0)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        distributed.barrier()
        distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        eps = 1e-6
        return self.total / (self.count + eps)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count > 0:
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print_func(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return distributed.get_world_size()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if distributed.get_backend() == 'nccl':
        return distributed.new_group(backend='gloo')

    return distributed.group.WORLD


def all_gather_cpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = 'cuda' if cpu_group is None else 'cpu'
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)]
    if cpu_group is None:
        distributed.all_gather(size_list, local_size)
    else:
        print('gathering on cpu')
        distributed.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        distributed.all_gather(tensor_list, tensor)
    else:
        distributed.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    if os.getenv('CPU_REDUCE') == '1':
        return all_gather_cpu(data)

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def collate_fn_for_cryo(batch):
    batch = list(zip(*batch))

    images = []
    for img_batch in batch[0]:
        images += img_batch
    batch[0] = nested_tensor_from_tensor_list(images)
    return tuple(batch)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f'sha: {sha}, status: {diff}, branch: {branch}'
    return message
