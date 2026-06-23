# emotion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

# --- 1. 모델 클래스 숨겨두기 ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s

class EffNetArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
        self.arcface = ArcMarginProduct(in_features=1408, out_features=num_classes)
    def forward(self, x): return self.arcface(self.backbone(x))

class ViTArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.arcface = ArcMarginProduct(in_features=self.backbone.num_features, out_features=num_classes)
    def forward(self, x): return self.arcface(self.backbone(x))

# --- 2. 래퍼(Wrapper) 클래스 ---
class EmotionPredictor:
    def __init__(self, ensemble_weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.idx_to_emotion = {0: 'Anger', 1: 'Happy', 2: 'Panic', 3: 'Sadness'}
        self.transform = A.Compose([
            A.Resize(260, 260),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # 모델 생성 및 가중치 로드
        self.eff_model = EffNetArcFace().to(self.device)
        self.vit_model = ViTArcFace().to(self.device)
        
        checkpoint = torch.load(ensemble_weight_path, map_location=self.device)
        self.eff_model.load_state_dict(checkpoint['eff_state_dict'])
        self.vit_model.load_state_dict(checkpoint['vit_state_dict'])
        
        self.eff_model.eval()
        self.vit_model.eval()
        
    def predict(self, image_rgb, temperature=20.0):
        # 4-Crop TTA, 앙상블, 사후 교정 등 복잡한 로직을 전부 여기서 처리
        transformed = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        
        ttas = [
            transformed,
            torch.flip(transformed, dims=[3]),
            F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273],
            torch.flip(F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273], dims=[3])
        ]
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            eff_out, vit_out = 0, 0
            for img in ttas:
                eff_out += self.eff_model(img)
                vit_out += self.vit_model(F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False))
            
            eff_logits = (eff_out / 4).cpu().numpy()
            vit_logits = (vit_out / 4).cpu().numpy()
            
        ensemble_logits = (eff_logits * 0.5) + (vit_logits * 0.5)
        ensemble_logits[:, 0] -= 2.1
        ensemble_logits[:, 2] += 1.05
        ensemble_logits[:, 3] += 0.42
        
        logits_tensor = torch.tensor(ensemble_logits[0]) / temperature
        probs = F.softmax(logits_tensor, dim=0).numpy() * 100
        
        return [{'emotion': self.idx_to_emotion[idx], 'probability': round(probs[idx], 2)} 
                for idx in np.argsort(probs)[::-1][:3]]