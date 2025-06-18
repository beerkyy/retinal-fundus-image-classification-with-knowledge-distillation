# ssl_pretraining.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from timm import create_model

class ContrastiveTransform:
    """Simple transformation for contrastive learning with two views"""
    def __init__(self, base_transform):
        self.base_transform = base_transform
        
    def __call__(self, x):
        # return two independently augmented views
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2

class SSLDataset(Dataset):
    """Dataset for self-supervised learning"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        # collect image file paths
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            view1, view2 = self.transform(image)
            return view1, view2
        return image, image

class SimCLR(nn.Module):
    """Simplified SimCLR model"""
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        # Remove classification head if ViT and stash feature dim
        if hasattr(self.encoder.model, 'head'):
            # stash the feature dim for later finetuning
            self.encoder.feature_dim = self.encoder.model.head.in_features
            self.encoder.model.head = nn.Identity()
        else:
            # default embedding dim for ViT if head absent
            self.encoder.feature_dim = 768
        # projection head
        feat_dim = self.encoder.feature_dim
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        # returns representation h and normalized projection z
        h = self.encoder(x)
        z = self.projection(h)
        z = F.normalize(z, dim=1)
        return h, z

def contrastive_loss(z_i, z_j, temperature=0.5):
    """NT-Xent loss for two sets of projections"""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    sim_matrix = torch.mm(z, z.t()) / temperature  # [2N,2N]
    # mask out self similarities
    diag_mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(diag_mask, -9e15)
    # positive pairs are off-diagonals at distance N
    pos = torch.cat([sim_matrix.diag(batch_size), sim_matrix.diag(-batch_size)], dim=0)
    # log-sum-exp over rows for denominator
    denom = torch.logsumexp(sim_matrix, dim=1)
    loss = - (pos - denom).mean()
    return loss


def pretrain_model(data_dir, model, epochs=30, batch_size=32, lr=1e-4, device='cuda'):  
    """Perform SSL pretraining and return pretrained encoder"""
    # augmentation pipeline
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cont_transform = ContrastiveTransform(base_transform)
    dataset = SSLDataset(data_dir, transform=cont_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    ssl_model = SimCLR(model).to(device)
    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=lr)
    for epoch in range(epochs):
        ssl_model.train()
        total_loss = 0.0
        for view1, view2 in tqdm(loader, desc=f"SSL Epoch {epoch+1}/{epochs}"):
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            _, z1 = ssl_model(view1)
            _, z2 = ssl_model(view2)
            loss = contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(ssl_model.encoder.state_dict(), "pretrained_encoder.pth")
    return ssl_model.encoder


def finetune_model(pretrained_encoder, train_loader, val_loader,
                   num_classes=4, epochs=10, lr=1e-4, device='cuda'):
    """Finetune pretrained encoder for classification"""
    model = pretrained_encoder.to(device)
    # determine feature dimension
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'feature_dim'):
        feat_dim = model.encoder.feature_dim
    elif hasattr(model, 'feature_dim'):
        feat_dim = model.feature_dim
    elif hasattr(model, 'num_features'):
        feat_dim = model.num_features
    elif hasattr(model.model, 'config') and hasattr(model.model.config, 'hidden_size'):
        feat_dim = model.model.config.hidden_size
    else:
        feat_dim = 768
    # attach classification head
    if hasattr(model, 'encoder'):  # SimCLR encoder
        model.encoder.model.head = nn.Linear(feat_dim, num_classes)
    else:
        model.model.head = nn.Linear(feat_dim, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_correct = train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Train Finetune {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val Finetune {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_finetuned_model.pth")
    model.load_state_dict(torch.load("best_finetuned_model.pth"))
    return model

