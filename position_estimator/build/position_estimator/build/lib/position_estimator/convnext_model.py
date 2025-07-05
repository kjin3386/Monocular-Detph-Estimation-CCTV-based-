#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class ConvNeXtPositionEstimator(nn.Module):
    """ConvNeXt-based trash classification and position refinement model"""
    
    def __init__(self, num_classes=6, backbone='convnext_tiny'):
        super().__init__()
        
        # ConvNeXt backbone for image features
        if backbone == 'convnext_tiny':
            from torchvision.models import convnext_tiny
            self.backbone = convnext_tiny(pretrained=True)
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classifier to get features
        self.backbone.classifier = nn.Identity()
        
        # Distance embedding
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Combined feature dimension
        combined_dim = feature_dim + 64
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Position refinement head (optional)
        self.position_refiner = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # x, y, z offset
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = [
            'plastic_bottle', 'can', 'paper', 
            'glass', 'organic', 'other'
        ]

    def preprocess_image(self, image, bbox):
        """Crop and preprocess image based on bounding box"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop image
        cropped = image[y1:y2, x1:x2]
        
        # Handle empty crop
        if cropped.size == 0:
            cropped = np.zeros((50, 50, 3), dtype=np.uint8)
        
        # Convert BGR to RGB if needed
        if len(cropped.shape) == 3 and cropped.shape[2] == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        return self.transform(cropped)

    def forward(self, image, distance):
        """Forward pass through the model"""
        # Extract image features
        img_features = self.backbone(image)  # (B, 768)
        
        # Distance embedding
        dist_features = self.distance_embedding(distance.unsqueeze(-1))  # (B, 64)
        
        # Combine features
        combined = torch.cat([img_features, dist_features], dim=1)  # (B, 832)
        
        # Classification
        class_logits = self.classifier(combined)
        
        # Position refinement
        position_offset = self.position_refiner(combined)
        
        return class_logits, position_offset

    def predict(self, image, distance):
        """Predict class and position refinement"""
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            if isinstance(distance, (int, float)):
                distance = torch.tensor([distance]).float()
            
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(distance.shape) == 0:
                distance = distance.unsqueeze(0)
            
            class_logits, position_offset = self.forward(image, distance)
            
            # Get predicted class
            class_probs = torch.softmax(class_logits, dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            confidence = class_probs[0, predicted_class].item()
            
            return {
                'class_id': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'position_offset': position_offset[0].cpu().numpy()
            }


class ModelInference:
    """Model inference wrapper for ROS2 integration"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = ConvNeXtPositionEstimator()
        
        # Load trained model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict_trash(self, image, bbox, distance):
        """Predict trash class and refined position"""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            processed_img = self.model.preprocess_image(image, bbox)
            processed_img = processed_img.to(self.device)
            
            # Convert distance to tensor
            distance_tensor = torch.tensor([distance], dtype=torch.float32).to(self.device)
            
            # Predict
            result = self.model.predict(processed_img, distance_tensor)
            
            return result
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return None