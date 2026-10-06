# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list) -> bool:
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    param_dict_type = getattr(args, 'param_dict_type', 'default')
    if param_dict_type not in ['default', 'ddetr_in_mmdet']:
        raise ValueError(
            f'param_dict_type {param_dict_type} should be default or ddetr_in_mmdet',
        )

    # by default
    if param_dict_type == 'default':
        param_dicts = [
            {
                'params': [
                    p for n, p in model_without_ddp.named_parameters()
                    if 'backbone' not in n and p.requires_grad
                ]
            },
            {
                'params': [
                    p for n, p in model_without_ddp.named_parameters()
                    if 'backbone' in n and p.requires_grad
                ],
                'lr': args.lr_backbone,
            }
        ]
        return param_dicts

    elif param_dict_type == 'ddetr_in_mmdet':
        param_dicts = [
            {
                'params':
                    [
                        p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names)
                        and not match_name_keywords(n, args.lr_linear_proj_names)
                        and p.requires_grad
                    ],
                'lr': args.lr,
            },
            {
                'params':
                    [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
                    ],
                'lr': args.lr_backbone,
            },
            {
                'params':
                    [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
                    ],
                'lr': args.lr_linear_proj_mult,
            }
        ]
        return param_dicts

    else:
        raise NotImplementedError
