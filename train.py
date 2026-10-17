# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import argparse
import collections
import copy
import datetime
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import fulltilt.engine as engine
import fulltilt.util.misc as misc
from fulltilt.datasets import build_dataset
from fulltilt.models import registry
from fulltilt.util.get_param_dicts import get_param_dict
from fulltilt.util.logger import setup_logger
from fulltilt.util.slconfig import DictAction, SLConfig


def get_args_parser():
    parser = argparse.ArgumentParser('FullTilt Training', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
    )

    # dataset parameters
    parser.add_argument('--datasets', type=str, required=True, help='path to datasets json')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument(
        '--dist_url', default='env://', help='url used to set up distributed training',
    )
    parser.add_argument('--rank', default=0, type=int, help='rank of distributed processes')
    parser.add_argument('--local-rank', type=int, help='local rank for DistributedDataParallel')

    # other parameters
    parser.add_argument('--amp', action='store_true', help='Train with mixed precision')
    parser.add_argument('--log_tensorboard', action='store_true', help='Store tensorboard logs')
    return parser


def build_model_main(args):
    if args.modelname not in registry.MODULE_BUILD_FUNCS._module_dict:
        raise ValueError(f'Model name {args.modelname} not found in registry')

    build_func = registry.MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, processor_3d = build_func(args, training=True)
    return model, criterion, processor_3d


def main(args):
    # ----------------------------------------
    # set up env
    # ----------------------------------------
    misc.setup_distributed(args)

    # load cfg file and update the args
    print(f'Loading config file from {args.config_file}')
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, 'config_cfg.py')
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, 'config_args_raw.json')
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError(f'Key {k} can used by args only')

    # ----------------------------------------
    # set up logger
    # ----------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'info.txt'),
        distributed_rank=args.rank,
        color=False,
        name='fulltilt',
    )

    logger.info(f'git:\n  {misc.get_sha()}\n')
    logger.info(f'Command: {" ".join(sys.argv)}')
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, 'config_args_all.json')
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f'Full config saved to {save_json_path}')

    logger.info(f'world size: {args.world_size}')
    logger.info(f'rank: {args.rank}')
    logger.info(f'local_rank: {args.local_rank}')
    logger.info(f'args: {args}\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------------------
    # set up tensorboard
    # ----------------------------------------
    writer = None
    if args.output_dir and misc.is_main_process() and args.log_tensorboard:
        writer = SummaryWriter(os.path.join(args.output_dir, './runs'))

    # ----------------------------------------
    # load datasets
    # ----------------------------------------
    with open(args.datasets) as f:
        dataset_meta = json.load(f)

    logger.debug('build dataset ... ...')
    dataset_train = None
    dataset_train_random = None
    if not args.eval:
        num_of_dataset_train = len(dataset_meta['train'])
        if num_of_dataset_train > 1:
            raise NotImplementedError('Not support num_of_dataset_train > 1')

        if num_of_dataset_train == 1:
            dataset_train = build_dataset(
                image_set='train', args=args, datasetinfo=dataset_meta['train'][0]
            )
            logger.debug('build dataset, done.')
            logger.debug(
                f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}'
            )

        if 'train_random' in dataset_meta:
            if len(dataset_meta['train_random']) != 1:
                raise NotImplementedError('Not support num_of_dataset_train_random != 1')
            dataset_train_random = build_dataset(
                image_set='train', args=args, datasetinfo=dataset_meta['train_random'][0]
            )
            logger.debug('build random dataset, done.')
            logger.debug(f'samples: {len(dataset_train_random)}')

    dataset_val_num = len(dataset_meta['val'])
    dataset_vals = [None] * dataset_val_num
    for dataset_idx, dataset_info in enumerate(dataset_meta['val']):
        dataset_vals[dataset_idx] = build_dataset(
            image_set='val', args=args, datasetinfo=dataset_info,
        )

    sampler_train = None
    sampler_train_random = None
    sampler_vals = [None] * dataset_val_num
    if args.distributed:
        for dataset_idx in range(dataset_val_num):
            sampler_vals[dataset_idx] = DistributedSampler(dataset_vals[dataset_idx], shuffle=False)

        if dataset_train is not None:
            sampler_train = DistributedSampler(dataset_train)

        if dataset_train_random is not None:
            sampler_train_random = DistributedSampler(dataset_train_random)

    else:
        for dataset_idx in range(dataset_val_num):
            sampler_vals[dataset_idx] = torch.utils.data.SequentialSampler(
                dataset_vals[dataset_idx]
            )

        if dataset_train is not None:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        if dataset_train_random is not None:
            sampler_train_random = torch.utils.data.RandomSampler(dataset_train_random)

    if dataset_train is not None:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=misc.collate_fn_for_cryo,
            num_workers=args.num_workers,
        )

    if dataset_train_random is not None:
        batch_sampler_train_random = torch.utils.data.BatchSampler(
            sampler_train_random, args.batch_size, drop_last=True)
        data_loader_train_random = DataLoader(
            dataset_train_random,
            batch_sampler=batch_sampler_train_random,
            collate_fn=misc.collate_fn_for_cryo,
            num_workers=args.num_workers,
        )

    data_loader_vals = [None] * dataset_val_num
    for dataset_idx in range(dataset_val_num):
        data_loader_vals[dataset_idx] = DataLoader(
            dataset_vals[dataset_idx],
            1,
            sampler=sampler_vals[dataset_idx],
            drop_last=False,
            collate_fn=misc.collate_fn_for_cryo,
            num_workers=args.num_workers,
        )

    # ----------------------------------------
    # set up model
    # ----------------------------------------
    logger.debug('build model ... ...')
    model, criterion, processor_3d = build_model_main(args)
    model.to(device)
    logger.debug('build model, done.')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params,
        )
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    logger.debug(
        f'params before freezing:\n'
        f'{json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2)}'  # noqa
    )

    param_dicts = get_param_dict(args, model_without_ddp)

    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break

    logger.debug(
        f'params after freezing:\n'
        f'{json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2)}'  # noqa
    )

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(data_loader_train),
            epochs=args.epochs,
            pct_start=0.2,
        )
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_drop_list,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # ----------------------------------------
    # load model weights
    # ----------------------------------------
    output_dir = pathlib.Path(args.output_dir)

    # NOTE: auto-resume from the last checkpoint if it exists, but override it if --resume
    # is explicitly specified.
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(misc.clean_state_dict(checkpoint['model']), strict=False)

        if (
            not args.eval
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        logger.info(f'Loading pre-trained model from {args.pretrain_model_path}')
        checkpoint = torch.load(
            args.pretrain_model_path, map_location='cpu', weights_only=False,
        )['model']
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info(f'Ignore keys: {json.dumps(ignorelist, indent=2)}')
        _tmp_st = collections.OrderedDict(
            {
                k: v for k, v in misc.clean_state_dict(checkpoint).items() if check_keep(
                    k, _ignorekeywordlist
                )
            }
        )

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    # ----------------------------------------
    # training
    # ----------------------------------------
    print('Start training')
    start_time = time.perf_counter()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()
        if args.distributed:
            if dataset_train is not None:
                sampler_train.set_epoch(epoch)
            if dataset_train_random is not None:
                sampler_train_random.set_epoch(epoch)

        if dataset_train_random is not None:
            train_random_stats = engine.train_one_epoch(
                model=model,
                criterion=criterion,
                processor_3d=processor_3d,
                data_loader=data_loader_train_random,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                lr_scheduler=lr_scheduler,
                args=args,
                logger=(logger if args.save_log else None),
            )

        if dataset_train is not None:
            train_stats = engine.train_one_epoch(
                model=model,
                criterion=criterion,
                processor_3d=processor_3d,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                lr_scheduler=lr_scheduler,
                args=args,
                logger=(logger if args.save_log else None),
            )

        if not args.onecyclelr:
            lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                misc.save_on_master(weights, checkpoint_path)

        # eval
        test_stats = [None] * dataset_val_num

        if (epoch + 1) % args.eval_interval == 0:
            print('eval...')
            for dataset_idx in range(dataset_val_num):
                test_stats_temp = engine.evaluate(
                    model=model,
                    criterion=criterion,
                    processor_3d=processor_3d,
                    data_loader=data_loader_vals[dataset_idx],
                    device=device,
                    args=args,
                    logger=(logger if args.save_log else None),
                )
                test_stats[dataset_idx] = copy.deepcopy(test_stats_temp)

        log_stats = {}
        if dataset_train is not None:
            log_stats.update({f'train_{k}': v for k, v in train_stats.items()})

        if dataset_train_random is not None:
            log_stats.update({f'trainrnd_{k}': v for k, v in train_random_stats.items()})

        for dataset_idx in range(dataset_val_num):
            if test_stats[dataset_idx] is not None:
                log_stats.update(
                    {f'test{dataset_idx + 1}_{k}': v for k, v in test_stats[dataset_idx].items()}
                )

        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except Exception:
            pass

        epoch_time = time.perf_counter() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and misc.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

            if args.log_tensorboard:
                for k, v in log_stats.items():
                    mode, item = k.split('_', 1)
                    if mode != 'train' and mode != 'trainrnd' and not mode.startswith('test'):
                        continue
                    item_name = f'{mode}/{item}'
                    writer.add_scalar(item_name, v, epoch + 1)

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'FullTilt training script', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
