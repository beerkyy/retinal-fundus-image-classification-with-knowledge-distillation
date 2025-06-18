#!/usr/bin/env python3
"""
Model evaluation module
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time


def evaluate_model(model, data_loader, criterion, device, class_names, print_report=True, plot_cm=True):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run evaluation on (cpu or cuda).
        class_names (list): List of class names for labeling reports/plots.
        print_report (bool): Whether to print classification report.
        plot_cm (bool): Whether to plot the confusion matrix.

    Returns:
        tuple: A tuple containing (average_loss, accuracy).
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store metrics and predictions
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    print(f"Starting evaluation on {len(data_loader.dataset)} samples...")
    
    # Disable gradient calculations for efficiency
    with torch.no_grad():
        # Iterate over the data loader
        for inputs, labels in data_loader:
            # Move data to the specified device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass: compute predictions
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0) # Accumulate loss scaled by batch size
            
            # Get predicted class indices
            _, preds = torch.max(outputs, 1)
            
            # Update total samples and correct predictions count
            total_samples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()
            
            # Store predictions and true labels for confusion matrix/report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    end_time = time.time()
    eval_duration = end_time - start_time
    
    # Print overall results
    print(f"Evaluation finished in {eval_duration:.2f} seconds.")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    # Optional: Print classification report
    if print_report and class_names:
        print("\nClassification Report:")
        # Ensure target_names matches the number of unique labels found
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        print(report)
    
    # Optional: Plot confusion matrix
    if plot_cm and class_names:
        cm = confusion_matrix(all_labels, all_preds)
        # Plot using seaborn for better visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        # Consider saving the plot instead of showing directly if running non-interactively
        plt.savefig("confusion_matrix.png") 
        #plt.show() 

    # Return the calculated metrics
    return average_loss, accuracy


            