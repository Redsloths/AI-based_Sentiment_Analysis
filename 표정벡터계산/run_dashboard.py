import fiftyone as fo
import pandas as pd
import os
from tqdm import tqdm

print("[단계 1] 시스템 설정 및 초기화")
CSV_PATH = "/workspace/user1/3_emotion_vector/complex_emotion_statistics_04.csv"
IMAGE_DIR = "/workspace/user1/dataset/open_images_faces/"
DATASET_NAME = "emotion_complex_v2_check"

print("[단계 2] CSV 데이터 로드")
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
except:
    df = pd.read_csv(CSV_PATH, encoding='cp949')
print(f" -> CSV 파일에서 {len(df)}줄을 읽어왔습니다.")

print("[단계 3] FiftyOne 데이터셋 DB 연결 (여기서 멈추면 좀비 프로세스 문제)")
if DATASET_NAME in fo.list_datasets():
    fo.delete_dataset(DATASET_NAME)
dataset = fo.Dataset(DATASET_NAME)

print("[단계 4] 이미지 물리 경로 매칭 및 샘플 생성")
samples = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="매칭 진행률"):
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    if not os.path.exists(img_path): continue
    
    sample = fo.Sample(filepath=img_path)
    sample["complex_emotion"] = fo.Classification(
        label=row['complex_label'],
        confidence=float(1 - row['distance'])
    )
    sample["distance"] = float(row['distance'])
    sample["anger_p"] = float(row['Anger_prob'])
    sample["happy_p"] = float(row['Happy_prob'])
    sample["panic_p"] = float(row['Panic_prob'])
    sample["sadness_p"] = float(row['Sadness_prob'])
    
    samples.append(sample)

print(f"[단계 5] DB에 {len(samples)}개 샘플 일괄 등록 (여기서 멈추면 메모리 부족)")
dataset.add_samples(samples)
dataset.persistent = True

print("[단계 6] 웹 대시보드 서버 가동")
# 포트 충돌 방지를 위해 5152로 변경 (필요 시 브라우저 접속 주소도 localhost:5152로 변경)
session = fo.launch_app(dataset, address="0.0.0.0", port=5152)

print("-" * 40)
print("★ 서버가 정상적으로 실행되었습니다. 브라우저에서 localhost:5152 로 접속하세요.")
print("★ 종료하려면 터미널에서 Ctrl+C 를 누르세요.")
print("-" * 40)

session.wait()