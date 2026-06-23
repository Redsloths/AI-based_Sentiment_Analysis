# 해상도 수정

import os
import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

# 사용자 정의 모듈 임포트
from emotion_model_02 import EmotionPredictor
from complex_emotion import EmotionAnchorSystem

def apply_clahe_color(image):
    """저조도 극장 환경을 위한 LAB 색상 공간 기반 CLAHE 전처리"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def run_pipeline():
    # 1. 경로 설정
    SOURCE_DIR = "/workspace/user1/3_emotion_vector/take_2/theater_pic/test/images" 
    OUTPUT_DIR = "/workspace/user1/3_emotion_vector/take_2/inference_results"
    
    YOLO_WEIGHTS = "/workspace/user1/3_emotion_vector/take_2/best.pt"
    EXCEL_PATH = "/workspace/user1/3_emotion_vector/take_2/감정조합임시_v3.xlsx"
    CNN_MODEL_PATH = "/workspace/user1/3_emotion_vector/take_2/final_ensemble_v1.pth"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 시스템 초기화
    print("🚀 시스템 초기화 중...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    predictor = EmotionPredictor(CNN_MODEL_PATH)
    emotion_system = EmotionAnchorSystem(EXCEL_PATH, expansion_k=0.3, device=device)
    
    target_emotions = ['Anger', 'Happy', 'Panic', 'Sadness']
    # CSV 컬럼 구조 정의
    columns = ['crop_id', 'bbox', 'complex_label', 'distance', 'Anger_prob', 'Happy_prob', 'Panic_prob', 'Sadness_prob', 'Neutral_prob']
    
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"총 {len(image_files)}장의 사진에 대해 파이프라인을 가동합니다.\n")

    # 3. 메인 추론 루프
    for img_name in tqdm(image_files, desc="전체 이미지 처리 진행률"):
        img_path = os.path.join(SOURCE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
            
        img_clahe = apply_clahe_color(original_img)
        
        # --- 1. YOLO 안면 탐지 (NMS 파라미터 튜닝) ---
        detect_results = yolo_model.predict(
            source=img_clahe, 
            conf=0.25,           # 0.22에서 노이즈가 안 잡혔으므로 0.25로 상향
            imgsz=1280,          
            iou=0.15,            # [극한 설정] 박스가 15%만 겹쳐도 무조건 하나를 삭제
            agnostic_nms=True,   
            save=False, 
            verbose=False
        )
        
        image_records = []
        
        for r in detect_results:
            for idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                if w < 36 or h < 36:
                    continue
                
                face_crop = img_clahe[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # CNN 모델 추론
                preds = predictor.predict(face_rgb)
                
                # --- 2. [핵심 추가] Neutral 확률 강제 가중치 부여 보정 로직 ---
                prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
                
                # Neutral에 1.8배 가중치를 주어 극장 관객의 디폴트 표정을 무표정으로 강제 견인
                NEUTRAL_WEIGHT = 1.8 
                prob_dict['Neutral'] = prob_dict.get('Neutral', 0.0) * NEUTRAL_WEIGHT
                
                # 가중치 부여로 인해 총합이 1.0을 넘었으므로, 다시 정규화(Normalization) 진행
                total_prob = sum(prob_dict.values())
                prob_dict = {k: v / total_prob for k, v in prob_dict.items()}
                
                # 확률 벡터 재조립 (정규화된 값 사용)
                probs_array = [[prob_dict.get(emo, 0.0) for emo in target_emotions + ['Neutral']]]
                probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device=device)
                
                # 복합 감정 엔진 연산
                final_labels, distances = emotion_system.classify(probs_tensor)
                
                # (이하 CSV 저장 로직은 기존과 동일)
                # ...
                
                image_records.append({
                    'crop_id': f"crop_{idx:03d}",
                    'bbox': f"[{x1}, {y1}, {x2}, {y2}]",
                    'complex_label': final_labels[0],
                    'distance': round(distances[0].item(), 4),
                    'Anger_prob': prob_dict.get('Anger', 0.0),
                    'Happy_prob': prob_dict.get('Happy', 0.0),
                    'Panic_prob': prob_dict.get('Panic', 0.0),
                    'Sadness_prob': prob_dict.get('Sadness', 0.0),
                    'Neutral_prob': prob_dict.get('Neutral', 0.0)
                })

        # 4. 개별 사진당 CSV 저장 (수정된 부분)
        if image_records:
            # 탐지된 얼굴이 있는 경우
            df = pd.DataFrame(image_records)
            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_emotions.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            # 탐지된 얼굴이 없는 경우: 빈 CSV 생성 및 파일명에 _blank 추가
            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_blank.csv")
            df_blank = pd.DataFrame(columns=columns)
            df_blank.to_csv(csv_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    run_pipeline()