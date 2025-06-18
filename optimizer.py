#!/usr/bin/env python3
"""
Optimizer and learning rate scheduler creation with layer-wise LR decay
"""
import math
import torch
import torch.optim as optim


def create_optimizer_and_scheduler(model, args):
    """Create optimizer with layer-wise LR decay and LR scheduler"""
    # Calculate effective batch size and scale LR
    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.blr * eff_batch_size / 256  # Scale base LR
    
    print(f"Effective batch size: {eff_batch_size}")
    print(f"Scaled learning rate (LR): {args.lr:.2e}")
    print(f"Base learning rate (BLR): {args.blr:.2e}")
    print(f"Minimum learning rate: {args.min_lr:.2e}")
    print(f"Layer decay: {args.layer_decay}")
    
    # Get parameter groups for LLRD
    param_groups = param_groups_lrd(
        model, 
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    
    # Apply LR scaling to each group based on its lr_scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * param_group.get('lr_scale', 1.0)
    
    # Create LR scheduler function
    lr_scheduler = lambda epoch: lr_lambda(epoch, args)
    
    return optimizer, lr_scheduler


def lr_lambda(current_epoch, args):
    """
    Calculate learning rate multiplier for each epoch
    Includes warmup and cosine decay
    """
    # Warmup phase
    if current_epoch < args.warmup_epochs:
        return float(current_epoch) / float(max(1, args.warmup_epochs))
    
    # Cosine decay phase
    else:
        progress = float(current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Ensure minimum LR is respected
        decayed_lr_ratio = (1.0 - args.min_lr / args.lr) * cosine_decay + (args.min_lr / args.lr)
        return decayed_lr_ratio


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=(), layer_decay=.75):
    """
    Parameter groups for layer-wise learning rate decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    
    # Assuming model.blocks is the list of transformer blocks
    num_layers = len(model.blocks) + 1  # +1 for patch_embed
    
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"
        
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    # Reformat to list of dicts for optimizer
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id for ViT models
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks.'):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:  # norm, head
        return num_layers