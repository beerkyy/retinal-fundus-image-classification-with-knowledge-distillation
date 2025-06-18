#!/usr/bin/env python3
"""
Utility functions for the fine-tuning process.
"""
import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The seed value to use.
    """
    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # Set seed for PyTorch CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU.
    # Ensure deterministic behavior for certain CUDA operations (can impact performance)
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False # Disable benchmark mode for determinism

def save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, args, best_acc=None, filename='checkpoint.pth'):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        lr_scheduler: The learning rate scheduler state.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler state.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Command-line arguments containing output_dir.
        best_acc (float, optional): The best validation accuracy achieved so far. Defaults to None.
        filename (str, optional): The name for the checkpoint file. Defaults to 'checkpoint.pth'.
    """
    # Prepare the state dictionary
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'args': args,
    }
    
    # Construct the full save path
    save_path = os.path.join(args.output_dir, filename)
    
    # Save the state dictionary to the specified path
    torch.save(save_state, save_path)
    print(f"Checkpoint saved to {save_path}")
