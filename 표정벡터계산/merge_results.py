import os
import glob
import pandas as pd

def merge_csv_by_timestamp():
    # 1. 경로 설정 (이전 스크립트의 출력 폴더)
    INPUT_DIR = "/workspace/user1/3_emotion_vector/take_2/inference_results"
    OUTPUT_PATH = "/workspace/user1/3_emotion_vector/take_2/inference_results/merged/merged_crowd_emotions.csv"
    
    # 해당 폴더 내의 모든 CSV 파일 검색
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not csv_files:
        print(f"❌ {INPUT_DIR} 경로에 병합할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 병합합니다...")
    
    df_list = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # 파일명에서 '_emotions.csv' 또는 '_blank.csv' 꼬리표 제거하여 타임스탬프/프레임명만 추출
            base_name = os.path.basename(file).replace('_emotions.csv', '').replace('_blank.csv', '')
            
            # 분석을 위해 데이터프레임의 맨 앞(0번째 열)에 타임스탬프 정보 삽입
            df.insert(0, 'source_frame', base_name)
            
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 ({file}): {e}")

    if df_list:
        # 2. 전체 데이터 병합 (Concat)
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # 3. 타임스탬프(파일명) 및 crop_id 기준으로 정렬하여 시계열 순서 확보
        if 'crop_id' in merged_df.columns:
            merged_df = merged_df.sort_values(by=['source_frame', 'crop_id'])
        else:
            merged_df = merged_df.sort_values(by=['source_frame'])
            
        # 4. 빈 탐지(_blank)의 빈 값(NaN) 처리
        merged_df = merged_df.fillna('None')

        # 5. 최종 저장
        merged_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        
        # 6. 간단한 통계 출력
        valid_faces = len(merged_df[merged_df['complex_label'] != 'None'])
        blank_frames = len(merged_df[merged_df['complex_label'] == 'None'])
        
        print(f"✅ 병합 완료: {OUTPUT_PATH}")
        print(f"📊 통계 요약: 유효 안면 {valid_faces}개 / 탐지 실패 프레임 {blank_frames}개")
    else:
        print("❌ 유효한 데이터프레임이 없습니다.")

if __name__ == "__main__":
    merge_csv_by_timestamp()