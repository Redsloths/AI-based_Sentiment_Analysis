import numpy as np
import pandas as pd
import torch

class EmotionAnchorSystem:
    def __init__(self, excel_path, expansion_k=0.3, device='cuda'):
        self.k = expansion_k
        self.device = device
        
        # 기본 5감정 앵커 (주의: 모델 출력 인덱스 순서와 반드시 일치시켜야 함)
        # 예시 순서: [Anger, Happy, Panic, Sadness, Neutral]
        self.base_anchors = np.array([
            [-0.7,  0.5], # 0: Anger
            [ 1.0,  0.1], # 1: Happy
            [-0.4,  0.9], # 2: Panic
            [-0.5, -0.6], # 3: Sadness
            [ 0.0,  0.0]  # 4: Neutral
        ], dtype=np.float32)
        
        self.base_anchors_tensor = torch.tensor(self.base_anchors, device=self.device)
        
        # 엑셀 데이터 로드 및 정답지(Complex Anchors) 생성
        self.complex_labels, self.complex_anchors = self._initialize_anchors(excel_path)
        # 명시적으로 dtype=torch.float32 지정
        self.complex_anchors_tensor = torch.tensor(self.complex_anchors, dtype=torch.float32, device=self.device)
        
    def _initialize_anchors(self, path):
        df = pd.read_excel(path, header=1).dropna(subset=['감정']).fillna(0)
        labels = df['감정'].tolist()
        
        # 엑셀 컬럼 역시 모델 출력 순서와 동일하게 정렬
        target_cols = ['Anger', 'Happy', 'Panic', 'Sadness', 'Neutral']
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        weights = df[target_cols].values
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums
        
        # NumPy로 정답지 좌표 변환
        transformed_coords = self._transform_logic(weights)
        return labels, transformed_coords

    def _transform_logic(self, weights):
        coords = np.dot(weights, self.base_anchors)
        r = np.sqrt(np.sum(coords**2, axis=1))
        theta = np.arctan2(coords[:, 1], coords[:, 0])
        
        r_expanded = np.power(r, self.k)
        
        # 1사분면 각도 분산 (15도 ~ 75도)
        mask_q1 = (theta > 0) & (theta < np.pi/2)
        if mask_q1.any():
            theta_q1 = theta[mask_q1]
            min_t, max_t = theta_q1.min(), theta_q1.max()
            if min_t != max_t:
                theta[mask_q1] = np.deg2rad(15) + (theta_q1 - min_t) * (np.deg2rad(75) - np.deg2rad(15)) / (max_t - min_t)
                
        x_new = r_expanded * np.cos(theta)
        y_new = r_expanded * np.sin(theta)
        return np.column_stack((x_new, y_new))

    def classify(self, model_probs):
        """
        model_probs: 모델의 Softmax 출력 텐서, shape: (Batch, 5)
        반환값: (예측된 복합 감정 라벨 리스트, 최소 거리 리스트)
        """
        with torch.no_grad():
            # 1. 투영 (Batch 연산)
            coords = torch.matmul(model_probs, self.base_anchors_tensor)
            
            # 2. 방사형 확장 (r^k)
            r = torch.sqrt(torch.sum(coords**2, dim=1))
            theta = torch.atan2(coords[:, 1], coords[:, 0])
            r_expanded = torch.pow(r, self.k)
            
            # GPU 텐서에서 1사분면 각도 분산 처리
            mask_q1 = (theta > 0) & (theta < 3.14159/2)
            if mask_q1.any():
                theta_q1 = theta[mask_q1]
                min_t, max_t = theta_q1.min(), theta_q1.max()
                if min_t != max_t:
                    target_min, target_max = 15.0 * 3.14159 / 180, 75.0 * 3.14159 / 180
                    theta[mask_q1] = target_min + (theta_q1 - min_t) * (target_max - target_min) / (max_t - min_t)
            
            x_new = r_expanded * torch.cos(theta)
            y_new = r_expanded * torch.sin(theta)
            transformed_preds = torch.stack((x_new, y_new), dim=1) # (Batch, 2)
            
            # 3. 최단 거리(유클리디안) 연산 (Batch 내 모든 데이터 vs 19개 앵커)
            # torch.cdist 반환 shape: (Batch, 19)
            distances = torch.cdist(transformed_preds, self.complex_anchors_tensor)
            
            # 4. 가장 짧은 거리를 가진 인덱스 추출
            min_dist, min_indices = torch.min(distances, dim=1)
            
            final_labels = [self.complex_labels[idx] for idx in min_indices.cpu().numpy()]
            return final_labels, min_dist.cpu().numpy()