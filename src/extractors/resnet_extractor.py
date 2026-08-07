import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ResNet18FeatureExtractor:
    def __init__(self, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize frozen, ImageNet-pretrained ResNet-18 model
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
        except ImportError:
            self.model = models.resnet18(pretrained=True)
            
        # Replace the final fc layer with nn.Identity
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Define exact preprocessing transformations
        # Gray -> 224x224 -> repeat to 3 channels -> normalize with ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Replicate single grayscale channel to 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # Normalization using standard ImageNet mean and std
            # Reused as a practical approximation, not recomputed for NIR/fingerprint distributions
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_path, l2_normalize=True):
        """
        Loads image as grayscale, runs preprocessing, and extracts 512-D pooled features.
        If l2_normalize is True, the output vector is L2-normalized to unit scale.
        """
        # Load image as grayscale (single channel)
        img = Image.open(image_path).convert('L')
        
        # Preprocess
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(tensor).squeeze(0).cpu().numpy()
            
        if l2_normalize:
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
                
        return features

    def extract_features_from_array(self, image_array, l2_normalize=True, target_size=None):
        """
        Accepts PIL Image or numpy array in-memory, runs preprocessing, and extracts 512-D features.
        If target_size is specified (e.g., (128, 512)), overrides default resize dynamically.
        """
        if isinstance(image_array, np.ndarray):
            img = Image.fromarray(image_array)
        else:
            img = image_array

        if img.mode != 'L':
            img = img.convert('L')

        if target_size is not None:
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = self.transform

        # Preprocess
        tensor = transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            features = self.model(tensor).squeeze(0).cpu().numpy()

        if l2_normalize:
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm

        return features
