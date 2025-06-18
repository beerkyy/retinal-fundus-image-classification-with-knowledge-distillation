#!/usr/bin/env python3
"""
Main entry point for ViT finetuning on fundus images
"""
import os
import torch
from pathlib import Path

from finetune_config import parse_args
from data_loading import create_dataloaders
from model import create_and_load_model, init_model_head
from optimizer import create_optimizer_and_scheduler
from trainer import train_and_evaluate
from utils import set_seed, save_checkpoint


def main():
    """Main function to run the training pipeline"""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup device and seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected! Training will be slow.")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes, class_names, class_weights_tensor = create_dataloaders(args)
    
    # Move class weights to the correct device *before* creating the criterion
    class_weights_tensor = class_weights_tensor.to(device)
    print(f"Using class weights on device: {class_weights_tensor.device}")

    # Update args with detected number of classes
    args.num_classes = num_classes
    print(f"Classes ({num_classes}): {class_names}")
    
    # Create and initialize model
    model = create_and_load_model(args, device)
    init_model_head(model)
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args)
    
    # Create loss function
    # Pass the device-specific weights to the loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing, weight=class_weights_tensor)
    print(f"Using Cross Entropy Loss with class weights and label smoothing: {args.smoothing}")
    # Note: LabelSmoothingCrossEntropy from timm might not directly accept weights.
    # If using args.smoothing > 0, you might need a custom implementation or check
    # if the specific timm version supports it. For simplicity, assuming standard CE here if weights are used.
    # if args.smoothing > 0:
    #     # from timm.loss import LabelSmoothingCrossEntropy
    #     # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) # May need modification for weights
    #     # print(f"Using Label Smoothing: {args.smoothing}")
    #     pass # Revisit if label smoothing + weights is needed
    # else:
    #     criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    #     print("Using standard Cross Entropy Loss with class weights")
    
    # Create grad scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Run training and evaluation
    best_state = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        device=device,
        args=args,
        class_names=class_names
    )
    
    # Final evaluation on test set
    if args.eval_test and best_state is not None:
        from evaluator import evaluate_model
        print("\nFinal evaluation on test set using best model...")
        model.load_state_dict(best_state)
        evaluate_model(model, test_loader, criterion, device, class_names)


if __name__ == "__main__":
    main()