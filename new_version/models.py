# This file contains loading, preprocessing, inspecting and batching up codes for retinal-fundus images for a 4-class classification problem 
# EE6892 Advanced Deep Learning on Edge Final Project
#
# Sources + custom implementation 
#
# @inproceedings{paszke2019pytorch,
#   author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and and Others},
#   title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
#   booktitle = {Advances in Neural Information Processing Systems 32},
#   pages     = {8024--8035},
#   year      = {2019}
# }
#
# @misc{torchvision,
#   author    = {{Torchvision} Contributors},
#   title     = {{Torchvision}: PyTorch’s Computer Vision Library},
#   howpublished = {\url{https://github.com/pytorch/vision}},
#   year      = {2020}
# }
#
# Requirements
# models.py

import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model

# The TeacherViT class wraps a Vision Transformer (ViT) model to serve as a “teacher” in a knowledge-distillation setup
# In a two-stage distillation, the teacher network is a large, pre-trained model whose internal representations guide 
# a smaller “student” network. Here, TeacherViT inherits from nn.Module, so it behaves just like any other PyTorch model: 
# we can move it to a device, call .train() or .eval(), and invoke it on input tensors
class TeacherViT(nn.Module):
    """ViT model for teacher network"""
    def __init__(self, num_classes=4, pretrained=True, model_name='vit_base_patch16_224'):
        super(TeacherViT, self).__init__()
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x, return_features=False):
        # Get the original forward function
        orig_forward = self.model.forward
        
        # Placeholder for features
        features = None
        
        # Hook function to capture intermediate features
        def hook_fn(module, inp, out):
            nonlocal features
            features = out
        
        hook_handle = None
        if return_features:
            # Register hook on the final transformer block
            if hasattr(self.model, 'blocks'):
                hook_handle = self.model.blocks[-1].register_forward_hook(hook_fn)
            else:
                hook_handle = self.model.transformer.encoder.layers[-1].register_forward_hook(hook_fn)
        
        # Forward pass
        logits = orig_forward(x)
        
        # Remove hook and return features if requested
        if return_features and hook_handle is not None:
            hook_handle.remove()
            return logits, features
        return logits

# The StudentCNN class defines a smaller convolutional-neural-network “student” that will learn both to classify and (optionally) to mimic the teacher’s internal representations.
# By inheriting from nn.Module, it plugs right into the PyTorch ecosystem: you can send it to GPU, wrap it in DataParallel or DistributedDataParallel, call .train()/.eval()
class StudentCNN(nn.Module):
    """CNN model for student network"""
    def __init__(self, num_classes=4, model_name='mobilenet_v2', pretrained=True):
        super(StudentCNN, self).__init__()
        self.model_name = model_name

        # Load the model with pretrained weights
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_layer = self.backbone.layer4
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_layer = self.backbone.layer4
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_layer = self.backbone.features
        elif model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.feature_layer = self.backbone.features
        elif model_name == 'squeezenet1_1':
            self.backbone = models.squeezenet1_1(pretrained=pretrained)
            self.feature_layer = self.backbone.features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Replace the final classification head
        if model_name.startswith('resnet'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'squeezenet1_1':
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

    def forward(self, x, return_features=False):
        # Placeholder for captured features
        features = None
        
        # Hook to capture features from the chosen layer
        def hook_fn(module, inp, out):
            nonlocal features
            features = out
        hook_handle = None
        if return_features:
            hook_handle = self.feature_layer.register_forward_hook(hook_fn)

        # Delegate forward to the backbone
        logits = self.backbone(x)

        # Remove hook and return features if requested
        if return_features and hook_handle is not None:
            hook_handle.remove()
            return logits, features
        return logits