#!/usr/bin/env python3
"""
Training and evaluation loop for model finetuning
"""
import os
import torch
from tqdm.auto import tqdm
from evaluator import evaluate_model
from utils import save_checkpoint


def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       criterion, optimizer, lr_scheduler, scaler, 
                       device, args, class_names):
    """Main training and evaluation loop"""
    print(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            device=device,
            args=args
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict().copy()
            
            # Save best model checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                accuracy=val_acc,
                args=args,
                class_names=class_names,
                is_best=True
            )
            
            print(f"New best model saved with accuracy: {best_val_acc:.2f}%")
        
        # Save periodic checkpoints
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                accuracy=val_acc,
                args=args,
                class_names=class_names,
                is_best=False
            )
    
    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    return best_state


def train_one_epoch(model, train_loader, criterion, optimizer, lr_scheduler, 
                    scaler, epoch, device, args):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0 # Accumulate total loss
    num_samples = 0 # Count total samples processed
    
    # Set learning rate for this epoch using scheduler
    current_lr_factor = lr_scheduler(epoch)
    for param_group in optimizer.param_groups:
        base_lr_scaled = args.lr * param_group.get('lr_scale', 1.0)
        param_group['lr'] = base_lr_scaled * current_lr_factor
    
    # Print current learning rate for the first group
    print(f"Epoch {epoch+1} | Set LR (group 0): {optimizer.param_groups[0]['lr']:.6e}")
    
    # Training loop
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for step, (inputs, labels) in progress_bar:
        # Get batch size
        batch_size = inputs.size(0)
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Normalize loss for gradient accumulation
        loss_normalized = loss / args.accum_iter
        
        # Backward pass with gradient scaling
        scaler.scale(loss_normalized).backward()
        
        # Optimizer step (with accumulation)
        if (step + 1) % args.accum_iter == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
        # Update accumulated loss and sample count
        # De-normalize loss for logging (multiply by accum_iter)
        current_loss = loss.item() 
        total_loss += current_loss * batch_size # Weighted by batch size
        num_samples += batch_size
        
        # Update progress bar
        progress_bar.set_postfix(
            loss=current_loss, # Show current batch loss
            avg_loss=total_loss / num_samples, # Show running average loss
            lr=optimizer.param_groups[0]['lr']
        )
    
    # Return the average loss for the epoch
    return total_loss / num_samples if num_samples > 0 else 0.0