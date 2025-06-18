# visualization.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import os

class GradCAM:
    """
    Grad-CAM implementation for visualizing model's decision
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model (nn.Module): Model to visualize
            target_layer: Target layer to compute Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM for a specific class
        
        Args:
            input_image (tensor): Input image
            target_class (int, optional): Target class to visualize. If None, uses the predicted class.
            
        Returns:
            tuple: (cam, output)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        target = torch.zeros_like(output)
        target[0, target_class] = 1
        output.backward(gradient=target, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence
        
        # Normalize CAM
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        cam = cam.squeeze().numpy()
        
        return cam, output

def apply_colormap(cam, img, alpha=0.5):
    """
    Apply colormap to CAM and overlay on image
    
    Args:
        cam (numpy.ndarray): Class activation map
        img (PIL.Image): Original image
        alpha (float): Transparency factor
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Resize CAM to match image dimensions
    cam = cv2.resize(cam, (img.width, img.height))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy array
    img_np = np.array(img)
    
    # Overlay heatmap on image
    cam_img = heatmap * alpha + img_np * (1 - alpha)
    cam_img = np.uint8(cam_img)
    
    return cam_img

class LayerRelevancePropagation:
    """
    Layer-wise Relevance Propagation (LRP) implementation
    """
    def __init__(self, model):
        """
        Args:
            model (nn.Module): Model to visualize
        """
        self.model = model
        self.model.eval()
        
    def generate_lrp(self, input_image, target_class=None, epsilon=1e-6):
        """
        Generate LRP for a specific class
        
        Args:
            input_image (tensor): Input image
            target_class (int, optional): Target class to visualize. If None, uses the predicted class.
            epsilon (float): Small constant for numerical stability
            
        Returns:
            tuple: (relevance_map, output)
        """
        # Forward pass
        input_image.requires_grad_(True)
        output = self.model(input_image)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Create one-hot encoding of the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass with LRP
        relevance = one_hot * output
        
        # Perform backward pass
        relevance.backward(retain_graph=True)
        
        # Get gradients
        relevance_map = input_image.grad.data.abs().sum(dim=1).squeeze().cpu().numpy()
        
        # Normalize relevance map
        relevance_map = relevance_map - relevance_map.min()
        relevance_map = relevance_map / (relevance_map.max() + epsilon)
        
        return relevance_map, output

def visualize_explanations(model, image_path, target_class=None, save_dir='./visualizations'):
    """
    Visualize model explanations using Grad-CAM and LRP
    
    Args:
        model (nn.Module): Model to visualize
        image_path (str): Path to input image
        target_class (int, optional): Target class to visualize. If None, uses the predicted class.
        save_dir (str): Directory to save visualizations
        
    Returns:
        tuple: (grad_cam_img, lrp_img, predicted_class, prediction_confidence)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Initialize Grad-CAM
    # Assuming a ResNet model structure, targeting the last convolutional layer
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer4'):
            target_layer = model.backbone.layer4[-1].conv2
        else:  # For EfficientNet
            target_layer = model.backbone.features[-1]
    else:
        # If direct model, try to find the last convolutional layer
        found = False
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                found = True
                break
        if not found:
            raise ValueError("Could not find a convolutional layer for Grad-CAM")
    
    grad_cam = GradCAM(model, target_layer)
    
    # Generate Grad-CAM
    cam, output = grad_cam.generate_cam(input_tensor, target_class)
    
    # Get predicted class and confidence
    predicted_class = output.argmax(dim=1).item()
    prediction_confidence = F.softmax(output, dim=1)[0, predicted_class].item()
    
    # Apply colormap to CAM and overlay on image
    grad_cam_img = apply_colormap(cam, img)
    
    # Generate LRP
    lrp = LayerRelevancePropagation(model)
    relevance_map, _ = lrp.generate_lrp(input_tensor, target_class)
    
    # Apply colormap to LRP and overlay on image
    lrp_img = apply_colormap(relevance_map, img)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grad_cam_img)
    plt.title(f'Grad-CAM (Class {predicted_class}, Conf: {prediction_confidence:.2f})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(lrp_img)
    plt.title(f'LRP (Class {predicted_class}, Conf: {prediction_confidence:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, os.path.basename(image_path).split('.')[0] + '_explanation.png'))
    plt.close()
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    return grad_cam_img, lrp_img, predicted_class, prediction_confidence