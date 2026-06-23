#!/usr/bin/env python
# coding: utf-8

# 코드 14
# ViT(Vision Transformer) + EfficientNet ensemble
# 앙상블 가중치 조절
# 과적합 문제로 lightGBM 제거. 가중치 0.5:0.5
# 

# In[ ]:


import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= 1. 환경 및 경로 설정 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EFF_WEIGHT_PATH = '/workspace/user1/model_v8_arcface/best_stage2_arcface.pth'
VIT_WEIGHT_PATH = '/workspace/user1/model_v8_arcface/vit_arcface_sync_best.pth'
IMG_PATH_INPUT = '/workspace/데이터셋/img/train/sadness/0h6a943225c61e4575d0cb3a9bb2de87914259f8c323c1a595992ad7d51e9lhw3.jpg'  # 테스트할 이미지 경로 입력

idx_to_emotion = {0: 'Anger', 1: 'Happy', 2: 'Panic', 3: 'Sadness'}

# 실전용 전처리 (EfficientNet 기준 260x260)
inference_transform = A.Compose([
    A.Resize(260, 260),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ================= 2. 모델 아키텍처 정의 =================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        # 실전 추론 시에는 코사인 유사도(각도)에 스케일만 곱해서 반환
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s

class EffNetArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super(EffNetArcFace, self).__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
        self.arcface = ArcMarginProduct(in_features=1408, out_features=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.arcface(features)

class ViTArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super(ViTArcFace, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        embed_dim = self.backbone.num_features 
        self.arcface = ArcMarginProduct(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.arcface(features)

# ================= 3. 가중치 로드 및 준비 =================
print("모델 가중치를 로드합니다...")
eff_model = EffNetArcFace(num_classes=4).to(DEVICE)
eff_model.load_state_dict(torch.load(EFF_WEIGHT_PATH, map_location=DEVICE))
eff_model.eval()

vit_model = ViTArcFace(num_classes=4).to(DEVICE)
vit_model.load_state_dict(torch.load(VIT_WEIGHT_PATH, map_location=DEVICE))
vit_model.eval()
print("✅ 가중치 로드 완료.")

# ================= 4. 최종 추론 파이프라인 =================
def predict_emotion(image_rgb, eff_model, vit_model, device, temperature=20.0):
    transformed = inference_transform(image=image_rgb)['image'].unsqueeze(0).to(device)
    
    # 4-Crop TTA 생성
    img_orig = transformed
    img_flip = torch.flip(transformed, dims=[3])
    
    n, c, h, w = transformed.shape
    img_zoom = F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)
    dy = (img_zoom.shape[2] - h) // 2
    dx = (img_zoom.shape[3] - w) // 2
    img_zoom = img_zoom[:, :, dy:dy+h, dx:dx+w]
    
    img_zoom_flip = torch.flip(img_zoom, dims=[3])
    
    ttas = [img_orig, img_flip, img_zoom, img_zoom_flip]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            eff_tta_out, vit_tta_out = 0, 0
            
            for tta_img in ttas:
                eff_tta_out += eff_model(tta_img)
                vit_img = F.interpolate(tta_img, size=(224, 224), mode='bilinear', align_corners=False)
                vit_tta_out += vit_model(vit_img)
                
            eff_logits = (eff_tta_out / len(ttas)).cpu().numpy()
            vit_logits = (vit_tta_out / len(ttas)).cpu().numpy()
            
    # 최적 가중치 앙상블 (0.5 : 0.5)
    ensemble_logits = (eff_logits * 0.5) + (vit_logits * 0.5)
    
    # 사후 교정 (Penalty: 2.1 적용)
    ensemble_logits[:, 0] -= 2.1 # anger
    ensemble_logits[:, 2] += 1.05 # panic
    ensemble_logits[:, 3] += 0.42 # sadness
    
    # Softmax + Temperature Scaling으로 확률 변환
    logits_tensor = torch.tensor(ensemble_logits[0]) / temperature
    probs = F.softmax(logits_tensor, dim=0).numpy() * 100
    
    sorted_indices = np.argsort(probs)[::-1]
    
    top3_result = []
    for i in range(3):
        idx = sorted_indices[i]
        top3_result.append({
            'emotion': idx_to_emotion[idx],
            'probability': round(probs[idx], 2)
        })
        
    return top3_result

# ================= 5. 실행 및 출력 테스트 =================
if __name__ == "__main__":
    img_path = IMG_PATH_INPUT  # 테스트할 이미지 경로 입력
    
    test_img = cv2.imread(img_path)
    if test_img is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
    else:
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        result = predict_emotion(test_img_rgb, eff_model, vit_model, DEVICE)
        
        print("\n" + "="*30)
        print(f"1) 최종 추론 감정 : {result[0]['emotion']}")
        print("="*30)
        print("2) Top-3 출력 결과")
        for rank, data in enumerate(result, 1):
            print(f"   {rank}위: {data['emotion']:7s} ({data['probability']}%)")
        print("="*30)

