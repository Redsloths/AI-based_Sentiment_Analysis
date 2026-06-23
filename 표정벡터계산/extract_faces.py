import cv2
import os
from tqdm import tqdm

def extract_faces_direct():
    # 1. 경로 설정
    SOURCE_DIR = "/home/user1/fiftyone/open-images-v7/validation/data/"
    SAVE_DIR = "/workspace/user1/dataset/open_images_faces/"
    LIMIT = 2000

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 2. OpenCV 내장 기본 얼굴 탐지기 로드
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 소스 디렉토리에서 이미지 파일 목록 가져오기
    try:
        image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.jpg')]
        print(f"소스 폴더에서 {len(image_files)}개의 이미지를 발견했습니다.")
    except FileNotFoundError:
        print(f"경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    # 3. 크롭 및 저장
    count = 0
    for filename in tqdm(image_files, desc="얼굴 크롭 진행률"):
        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # 탐지 정확도를 위해 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 탐지 수행
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60) # 너무 작은 노이즈 얼굴 제외
        )

        for (x, y, w, h) in faces:
            # 여백(Margin)을 살짝 주어 크롭 (선택 사항, 여기서는 딱 맞게 자름)
            face_img = img[y:y+h, x:x+w]
            
            if face_img.size > 0:
                save_path = os.path.join(SAVE_DIR, f"pure_face_{count:05d}.jpg")
                cv2.imwrite(save_path, face_img)
                count += 1

            if count >= LIMIT:
                break
                
        if count >= LIMIT:
            break

    print("-" * 30)
    print(f"★ 추출 완료: 총 {count}장의 얼굴 이미지가 '{SAVE_DIR}'에 저장되었습니다.")

if __name__ == "__main__":
    extract_faces_direct()