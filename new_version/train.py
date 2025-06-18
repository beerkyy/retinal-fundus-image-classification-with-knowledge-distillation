import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import time
import os

# Import the CrossArchitectureKD wrapper
from cross_kd2 import CrossArchitectureKD

def train_student_with_kd(teacher, student, dataloaders, num_classes=4, epochs=100, 
                         lr=0.001, weight_decay=1e-4, device='cuda', save_dir='./models'):
    """
    Train student model with knowledge distillation
    
    Args:
        teacher (nn.Module): Teacher model (ViT)
        student (nn.Module): Student model (CNN)
        dataloaders (dict): DataLoaders for training and validation
        num_classes (int): Number of output classes
        epochs (int): Number of epochs to train
        lr (float): Learning rate
        weight_decay (float): Weight decay for optimizer
        device (str): Device to train on
        save_dir (str): Directory to save model checkpoints
        
    Returns:
        student (nn.Module): Trained student model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Move models to device
    teacher = teacher.to(device)
    student = student.to(device)
    
    # Set teacher to evaluation mode
    teacher.eval()
    
    # Initialize cross-architecture KD wrapper
    kd_model = CrossArchitectureKD(
        teacher=teacher,
        student=student,
        feature_dim=768,  # ViT feature dimension
        num_views=2,
        lambda_robust=0.1
    ).to(device)
    
    # Initialize optimizer: include student and all KD components
    optimizer = optim.AdamW(
        [
            {'params': student.parameters()},
            {'params': kd_model.pca_projector.parameters()},
            {'params': kd_model.gl_projector.parameters()},
            {'params': kd_model.discriminator.parameters()}
        ],
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Classification loss
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training phase
        student.train()
        kd_model.train()
        
        train_loss = train_ce = train_kd = 0.0
        train_correct = train_total = 0
        
        train_loader = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward student
            outputs = student(inputs)
            classification_loss = ce_loss(outputs, targets)
            
            # KD loss on raw inputs
            kd_total_loss, _, _, _ = kd_model.calculate_kd_loss(inputs)
            
            # Combine losses
            alpha = 0.7
            loss = (1 - alpha) * classification_loss + alpha * kd_total_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_ce += classification_loss.item() * batch_size
            train_kd += kd_total_loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            train_total += batch_size
            train_correct += (preds == targets).sum().item()
            
            train_loader.set_postfix({
                'loss': loss.item(),
                'ce': classification_loss.item(),
                'kd': kd_total_loss.item(),
                'acc': 100 * (preds == targets).sum().item() / batch_size
            })
        
        # Validation phase
        student.eval()
        kd_model.eval()
        val_loss = val_correct = val_total = 0
        all_targets, all_preds, all_probs = [], [], []
        
        val_loader = tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                loss = ce_loss(outputs, targets)
                
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                val_total += batch_size
                val_correct += (preds == targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                
                val_loader.set_postfix({'loss': loss.item(), 'acc': 100 * (preds == targets).sum().item() / batch_size})
        
        # Compute epoch metrics
        train_loss /= len(dataloaders['train'].dataset)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(dataloaders['val'].dataset)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, " \
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best student
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(save_dir, "best_student_model.pth"))
            print(f"New best model saved: Val Acc {val_acc:.2f}%")
            
            # Detailed metrics
            target_names = [f"Class {i}" for i in range(num_classes)]
            print(classification_report(all_targets, all_preds, target_names=target_names))
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
            plt.close()
        
    # Plot curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()
    
    # Load and evaluate
    student.load_state_dict(torch.load(os.path.join(save_dir, "best_student_model.pth")))
    evaluate_model(student, dataloaders['test'], num_classes, device, save_dir)
    return student


def evaluate_model(model, dataloader, num_classes=4, device='cuda', save_dir='./models'):
    """
    Evaluate model on test set
    """
    model.eval()
    test_loss = test_correct = test_total = 0
    all_targets, all_preds, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            test_total += batch_size
            test_correct += (preds == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    test_loss /= len(dataloader.dataset)
    test_acc = 100 * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Report and plots
    target_names = [f"Class {i}" for i in range(num_classes)]
    print(classification_report(all_targets, all_preds, target_names=target_names))
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Test Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()
    
    # ROC curves
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    plt.figure(figsize=(8,6))
    for i in range(num_classes):
        y_true = (all_targets == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curves')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()
    return test_acc
