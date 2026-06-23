import os
import glob
import ast
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_from_csv_folder():
    # 1. 경로 설정
    CSV_DIR = "/workspace/user1/3_emotion_vector/take_2/inference_results"  # 개별 CSV들이 있는 폴더
    IMAGE_DIR = "/workspace/user1/3_emotion_vector/take_2/theater_pic/test/images" # 원본 테스트 사진 폴더
    OUTPUT_DIR = "/workspace/user1/3_emotion_vector/take_2/visualize" # 그림이 그려진 결과물 저장 폴더

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 폴더 내 모든 CSV 파일 스캔
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ {CSV_DIR} 경로에 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 순회하며 시각화를 시작합니다...\n")

    # 한글 폰트 설정 (리눅스 환경)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    processed_count = 0

    # 2. CSV 파일 순회 루프
    for csv_path in csv_files:
        # 파일명에서 원본 이미지 이름 추출
        base_name = os.path.basename(csv_path)
        if base_name.endswith('_emotions.csv'):
            img_name = base_name.replace('_emotions.csv', '')
            is_blank = False
        elif base_name.endswith('_blank.csv'):
            img_name = base_name.replace('_blank.csv', '')
            is_blank = True
        else:
            continue # 규격 외 파일 무시

        # 원본 이미지 찾기 (.jpg 또는 .png)
        img_path = os.path.join(IMAGE_DIR, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, f"{img_name}.png")
            if not os.path.exists(img_path):
                print(f"⚠️ 매칭되는 원본 이미지 없음, 건너뜀: {img_name}")
                continue

        # 이미지 로드 및 PIL 변환
        img_cv = cv2.imread(img_path)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 3. 데이터에 따른 BBox 작도
        if is_blank:
            # 얼굴 탐지 실패 이미지: 좌측 상단에 붉은색 경고 표시
            draw.text((10, 10), "Face Not Detected (0 BBox)", font=font, fill="red")
        else:
            # 얼굴이 탐지된 이미지: CSV 데이터를 읽어 박스 그리기
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                if pd.isna(row.get('bbox')) or row.get('complex_label') == 'None':
                    continue

                try:
                    box = ast.literal_eval(row['bbox'])
                    x1, y1, x2, y2 = box
                except (ValueError, SyntaxError):
                    continue
                
                label_name = row.get('complex_label', 'None')
                
                # --- [수정] 감정별 BBox 색상 매핑 (Hex Code) ---
                color_map = {
                    '기쁨': '#FDE047',   # 노란색
                    '편안': '#86EFAC',   # 연두색
                    '무표정': '#9CA3AF', # 회색
                    '슬픔': '#93C5FD',   # 파란색
                    '분노': '#FCA5A5',   # 빨간색
                    '경멸': '#F87171',   # 진한 빨간색
                    '공포': '#C084FC',   # 보라색
                    '혼란': '#FDBA74',   # 주황색
                    '염려': '#FCD34D',   # 짙은 노란색
                    '실망': '#6B7280'    # 진한 회색
                }
                
                # 라벨에 해당하는 색상이 없으면 기본 하늘색(#22d3ee) 적용
                box_color = color_map.get(label_name, "#22d3ee")
                label_text = f"{label_name} ({row.get('distance', 0.0):.2f})"

                # BBox 외곽선 (지정된 색상 적용)
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                
                # 텍스트 가독성용 배경 박스 (지정된 색상 적용)
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                draw.rectangle([x1, y1 - 20, x1 + (text_bbox[2] - text_bbox[0]) + 10, y1], fill=box_color)
                
                # 감정 텍스트 (글씨는 검은색 고정)
                draw.text((x1 + 5, y1 - 18), label_text, font=font, fill="black")

        # 4. 결과 저장
        res_img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(OUTPUT_DIR, f"checked_{img_name}.jpg")
        cv2.imwrite(save_path, res_img_cv)
        processed_count += 1

    print(f"✅ 총 {processed_count}장의 이미지 시각화가 완료되었습니다. ({OUTPUT_DIR})")

if __name__ == "__main__":
    visualize_from_csv_folder()