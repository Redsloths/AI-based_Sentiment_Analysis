# predict.py
import cv2
from emotion_model import EmotionPredictor

# 1. 모델 로드 (가중치 파일 지정)
model = EmotionPredictor("final_ensemble_v1.pth")

# 2. 원본 이미지 로드 및 전처리
original_image = cv2.imread("test img 3..png")
if original_image is None:
    raise FileNotFoundError("test img.png 파일을 찾을 수 없습니다.")

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 3. 모델 예측 수행
preds = model.predict(image)

# 4. 전체 Top-3 결과 확인
print("--- 추론 결과 (Top 3) ---")
for rank, data in enumerate(preds, 1):
    print(f"{rank}위: {data['emotion']:7s} ({data['probability']}%)")

# 5. 가장 높은 확률을 가진 1등 감정 클래스만 딱 추출하기
# (preds 리스트의 첫 번째[0] 요소의 'emotion' 값을 가져오면 됩니다)
final_class = preds[0]['emotion']
final_prob = preds[0]['probability']

print("-" * 25)
print(f"★ 최종 예측 감정: {final_class} ({final_prob}%)")