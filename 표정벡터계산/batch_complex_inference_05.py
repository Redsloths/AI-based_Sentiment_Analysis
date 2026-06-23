import os
import shutil
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

    # [수정] 이전 CSV 찌꺼기 완벽 삭제를 통한 데이터 중복 병합 방지
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 시스템 초기화
    yolo_model = YOLO(YOLO_WEIGHTS)
    predictor = EmotionPredictor(CNN_MODEL_PATH)
    emotion_system = EmotionAnchorSystem(EXCEL_PATH, expansion_k=0.3, device=device)
    
    # 모델 출력 4감정 + 추가할 Neutral
    target_emotions = ['Anger', 'Happy', 'Panic', 'Sadness']
    columns = ['crop_id', 'bbox', 'complex_label', 'distance', 'Anger_prob', 'Happy_prob', 'Panic_prob', 'Sadness_prob', 'Neutral_prob']
    
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 3. 메인 추론 루프
    for img_name in tqdm(image_files, desc="전체 이미지 처리 진행률"):
        img_path = os.path.join(SOURCE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
            
        img_clahe = apply_clahe_color(original_img)
        
        # [수정] YOLO 안면 탐지: NMS 극한 통제 및 conf 타협점 적용
        detect_results = yolo_model.predict(
            source=img_clahe, 
            conf=0.25,           
            imgsz=1280,          
            iou=0.15,            # 박스가 15%만 겹쳐도 무조건 하나를 삭제
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
                
                # CNN 감정 분류기 추론 (4개 클래스만 반환됨)
                preds = predictor.predict(face_rgb)
                
                # --- [핵심 수정] 도메인 캘리브레이션 (잔여 확률 전환 기법) ---
                prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
                
                # 어두운 조명과 그림자로 인해 발생하는 '찡그림(분노)'과 '우울(슬픔)' 오탐지를 강제 삭감
                PENALTY_ANGER = 0.3   # 분노 확률의 70% 삭감 (30%만 남김)
                PENALTY_SADNESS = 0.4 # 슬픔 확률의 60% 삭감 (40%만 남김)
                
                prob_dict['Anger'] = prob_dict.get('Anger', 0.0) * PENALTY_ANGER
                prob_dict['Sadness'] = prob_dict.get('Sadness', 0.0) * PENALTY_SADNESS
                
                # 삭감 후 남은 확률의 총합 계산
                current_sum = sum(prob_dict.values())
                
                # 날아간 빈 공간을 극장 관객의 디폴트 상태인 'Neutral(무표정)'으로 채움
                prob_dict['Neutral'] = max(0.0, 1.0 - current_sum)
                
                # 확률 벡터 재조립 (엑셀 엔진의 5차원 입력 규격에 맞춤)
                probs_array = [[
                    prob_dict.get('Anger', 0.0),
                    prob_dict.get('Happy', 0.0),
                    prob_dict.get('Panic', 0.0),
                    prob_dict.get('Sadness', 0.0),
                    prob_dict.get('Neutral', 0.0)
                ]]
                probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device=device)
                
                # 복합 감정 엔진 연산
                final_labels, distances = emotion_system.classify(probs_tensor)
                
                image_records.append({
                    'crop_id': f"crop_{idx:03d}",
                    'bbox': f"[{x1}, {y1}, {x2}, {y2}]",
                    'complex_label': final_labels[0],
                    'distance': round(distances[0].item(), 4),
                    'Anger_prob': round(prob_dict.get('Anger', 0.0), 4),
                    'Happy_prob': round(prob_dict.get('Happy', 0.0), 4),
                    'Panic_prob': round(prob_dict.get('Panic', 0.0), 4),
                    'Sadness_prob': round(prob_dict.get('Sadness', 0.0), 4),
                    'Neutral_prob': round(prob_dict.get('Neutral', 0.0), 4)
                })

        # 4. 개별 사진당 CSV 저장
        if image_records:
            df = pd.DataFrame(image_records)
            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_emotions.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_blank.csv")
            # 행(Row) 데이터로 'None' 값을 명시적으로 1줄 삽입
            df_blank = pd.DataFrame([['None'] * len(columns)], columns=columns)
            df_blank['crop_id'] = 'No_Face'
            df_blank.to_csv(csv_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    run_pipeline()