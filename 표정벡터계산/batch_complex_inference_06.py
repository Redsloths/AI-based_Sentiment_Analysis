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


def calculate_iou(box1, box2):
    """두 BBox 간의 겹침(IoU) 비율을 계산하는 함수"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

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
        
        # YOLO 안면 탐지 (CLAHE 이미지 사용)
        detect_results = yolo_model.predict(
            source=img_clahe, 
            conf=0.25,           
            imgsz=1280,          
            iou=0.15,            
            agnostic_nms=True,   
            save=False, 
            verbose=False
        )
        
        image_records = []
        accepted_boxes = [] # 수동 NMS를 위한 통과된 박스 리스트
        
        for r in detect_results:
            # YOLO가 내뱉은 결과를 confidence(확신도) 높은 순으로 정렬 보장
            sorted_boxes = sorted(r.boxes, key=lambda b: b.conf[0].item(), reverse=True)
            
            for idx, box in enumerate(sorted_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # 1. 크기 필터링
                if w < 36 or h < 36:
                    continue
                
                # 2. [핵심] 수동 NMS 적용: 기존에 통과된 박스와 20% 이상 겹치면 버림
                is_duplicate = False
                for ab in accepted_boxes:
                    if calculate_iou([x1, y1, x2, y2], ab) > 0.20:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                
                # 통과된 박스 등록
                accepted_boxes.append([x1, y1, x2, y2])
                
                # 3. [핵심] CNN 감정 분석용 이미지는 CLAHE가 아닌 '원본 이미지(original_img)'에서 크롭!
                face_crop = original_img[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # CNN 모델 추론
                preds = predictor.predict(face_rgb)
                
                # 도메인 캘리브레이션 (잔여 확률 전환 유지)
                prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
                
                # 원본 이미지를 쓰므로 페널티 수치를 살짝 완화 (필요시 조정 가능)
                PENALTY_ANGER = 0.5   
                PENALTY_SADNESS = 0.5 
                
                prob_dict['Anger'] = prob_dict.get('Anger', 0.0) * PENALTY_ANGER
                prob_dict['Sadness'] = prob_dict.get('Sadness', 0.0) * PENALTY_SADNESS
                
                current_sum = sum(prob_dict.values())
                prob_dict['Neutral'] = max(0.0, 1.0 - current_sum)
                
                probs_array = [[
                    prob_dict.get('Anger', 0.0),
                    prob_dict.get('Happy', 0.0),
                    prob_dict.get('Panic', 0.0),
                    prob_dict.get('Sadness', 0.0),
                    prob_dict.get('Neutral', 0.0)
                ]]
                probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device=device)
                
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