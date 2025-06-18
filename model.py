#!/usr/bin/env python3
"""
Model creation, initialization, and loading pretrained weights
"""
import os
import torch
import torch.nn as nn
from timm.models import create_model
from timm.layers import trunc_normal_


def create_and_load_model(args, device):
    """Create model and load pretrained weights"""
    print(f"Creating model: {args.model_name}")
    
    # Create model with specified parameters
    model = create_model(
        args.model_name,
        pretrained=False,  # We load weights manually
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        global_pool='avg' if args.global_pool else 'token',
    )
    
    # Load pretrained weights if checkpoint path is provided
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        load_pretrained_weights(model, args.checkpoint_path)
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint_path}. Training from scratch!")
    
    # Move model to device
    model.to(device)
    return model


def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights from checkpoint file"""
    print(f"Loading weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    # Handle different checkpoint formats (MAE, etc.)
    checkpoint_model = checkpoint.get('model', checkpoint)
    
    # Handle potential state_dict nesting in some models
    if 'state_dict' in checkpoint_model:
        checkpoint_model = checkpoint_model['state_dict']
    
    state_dict = model.state_dict()
    
    # Check for and remove mismatched head keys (classifier)
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from checkpoint due to shape mismatch")
            del checkpoint_model[k]
        elif k not in checkpoint_model:
            print(f"Key {k} not found in checkpoint, will use initialized head")
    
    # Interpolate position embedding if needed (size mismatch)
    if 'pos_embed' in checkpoint_model:
        interpolate_pos_embed(model, checkpoint_model)
    
    # Load weights with potential missing keys (strict=False)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Weight loading message:", msg)
    
    return model


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings if needed for different resolutions"""
    pos_embed_ckpt = checkpoint_model['pos_embed']
    embedding_size = pos_embed_ckpt.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0 for GAP, 1 for CLS token
    
    # Calculate original and new grid sizes
    orig_size = int((pos_embed_ckpt.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    
    if orig_size != new_size:
        print(f"Interpolating position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
        
        # Handle class token and position embeddings separately
        extra_tokens = pos_embed_ckpt[:, :num_extra_tokens]
        
        # Only interpolate position tokens
        pos_tokens = pos_embed_ckpt[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        
        # Use bicubic interpolation for position embeddings
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        
        # Update checkpoint with interpolated position embeddings
        checkpoint_model['pos_embed'] = new_pos_embed


def init_model_head(model):
    """Initialize the classification head"""
    print("Initializing classification head weights")
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.constant_(model.head.bias, 0)
    return model