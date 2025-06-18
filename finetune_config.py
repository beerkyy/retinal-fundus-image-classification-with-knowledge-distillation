#!/usr/bin/env python3
"""
Configuration module for ViT finetuning
"""
import argparse


def parse_args():
    """Parse command line arguments for the finetuning script"""
    parser = argparse.ArgumentParser('Fundus ViT Fine-tuning for Classification')
    
    # Model parameters
    parser.add_argument('--model_name', default='vit_large_patch16', type=str,
                        help="Model architecture to use")
    parser.add_argument('--num_classes', default=None, type=int, 
                        help="Number of classes (will be auto-detected if None)")
    parser.add_argument('--global_pool', action='store_true', default=True,
                        help="Use global average pooling")
    parser.add_argument('--use_cls_token', dest='global_pool', action='store_false',
                        help="Use CLS token for classification")
    parser.add_argument('--input_size', default=224, type=int, 
                        help="Input image size")
    
    # Data parameters
    parser.add_argument('--data_path', default='./data/', type=str, 
                        help="Path to dataset with train/val/test folders")
    parser.add_argument('--checkpoint_path', default='./pretrained_ViT/RETFound_cfp_weights.pth', type=str, 
                        help="Path to pre-trained MAE weights")
    parser.add_argument('--output_dir', default='./finetune_output', type=str,
                        help="Directory to save output files")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size per GPU")
    parser.add_argument('--epochs', default=50, type=int,
                        help="Number of epochs to train for")
    parser.add_argument('--blr', default=5e-3, type=float, 
                        help="Base learning rate (scaled by batch size)")
    parser.add_argument('--layer_decay', default=0.65, type=float,
                        help="Layer-wise learning rate decay factor")
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help="Weight decay for optimizer")
    parser.add_argument('--drop_path', default=0.2, type=float,
                        help="Stochastic depth rate")
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help="Label smoothing factor (0 to disable)")
    
    # Learning rate schedule parameters
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help="Number of epochs for warmup")
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help="Minimum learning rate")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Number of iterations for gradient accumulation")
    
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.2, 
                        help="Color jitter factor for fundus images")
    parser.add_argument('--auto_augment', action='store_true', 
                        help="Use AutoAugment for fundus images")
    
    # System parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device to use (cuda or cpu)")
    parser.add_argument('--seed', default=42, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of data loading workers")
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help="Pin memory in dataloader")
    
    # Evaluation parameters
    parser.add_argument('--eval_test', action='store_true',
                        help="Evaluate on test set after training")
    parser.add_argument('--save_freq', default=10, type=int,
                        help="Frequency of saving checkpoints (epochs)")
    
    return parser.parse_args()