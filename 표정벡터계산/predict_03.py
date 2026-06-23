import cv2
import warnings
import torch
from emotion_model_02 import EmotionPredictor
from complex_emotion import EmotionAnchorSystem 

# 불필요한 경고 숨김
warnings.filterwarnings("ignore")

def main():
    # 1. 모델 및 복합 감정 시스템 초기화
    predictor = EmotionPredictor("/workspace/user1/3_emotion_vector/take_1/final_ensemble_v1.pth")
    # A40 서버 환경에 맞게 'cuda' 명시
    emotion_system = EmotionAnchorSystem("/workspace/user1/3_emotion_vector/감정조합임시_v3.xlsx", expansion_k=0.3, device='cuda')

    # 2. 이미지 로드 및 전처리
    original_image = cv2.imread("/workspace/user1/3_emotion_vector/test_face.jpg")
    if original_image is None:
        raise FileNotFoundError("test_face.jpg 파일을 찾을 수 없습니다.")
    
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 3. 기존 모델 추론 (리스트 형태 반환: [{'emotion': 'Happy', 'probability': 85.0}, ...])
    preds = predictor.predict(image_rgb)
    
    if not preds:
        print("얼굴을 찾을 수 없거나 추론에 실패했습니다.")
        return

    # 4. 반환된 리스트를 5차원 확률 텐서로 재조립 (Re-vectorization)
    # 주의: complex_emotion.py의 base_anchors 배열 순서와 반드시 일치해야 함
    target_emotions = ['Anger', 'Happy', 'Panic', 'Sadness', 'Neutral']
    
    # 퍼센트(%)를 0~1 사이의 확률값으로 변환하여 딕셔너리로 매핑
    prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
    
    # 정해진 순서대로 확률값을 추출하여 [1, 5] 배열 생성 (없는 감정은 0.0 처리)
    probs_array = [[prob_dict.get(emo, 0.0) for emo in target_emotions]]
    
    # GPU 텐서로 변환
    probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device='cuda')

    # 5. 복합 감정 판별 엔진 가동
    final_labels, distances = emotion_system.classify(probs_tensor)

    # 6. 결과 출력
    print("--- 복합 감정 엔진 추론 결과 ---")
    print(f"★ 최종 판별 감정: {final_labels[0]}")
    print(f"기준 앵커와의 거리: {distances[0]:.4f}")
    
    # 분석용 로그 (재조립된 5차원 확률 분포)
    print("\n[내부 연산용 확률 벡터]")
    for emo, p in zip(target_emotions, probs_array[0]):
        print(f"- {emo:7s}: {p*100:5.2f}%")

if __name__ == "__main__":
    main()