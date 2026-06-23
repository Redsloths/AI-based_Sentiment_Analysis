import pandas as pd
import numpy as np

def calculate_scene_representative_emotion(file_path):
    # 1. 씬 전체 데이터 로드 및 정제
    df = pd.read_csv(file_path)
    valid_df = df[(df['crop_id'] != 'No_Face') & (df['complex_label'].notna())].copy()
    
    # 거리(가중치) 및 확률 컬럼 숫자형 변환
    prob_cols = ['Anger_prob', 'Happy_prob', 'Panic_prob', 'Sadness_prob', 'Neutral_prob']
    for col in prob_cols + ['distance']:
        valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')
    
    valid_df = valid_df.dropna(subset=prob_cols + ['distance'])

    def get_weighted_trimmed_mean(data, prob_col, weight_col='distance', proportion=0.1):
        """특정 확률값의 상하위 10%를 절사한 후, 남은 데이터에 대해 거리(distance) 가중 평균을 구함"""
        if len(data) == 0: return 0.0
        
        # 확률값을 기준으로 오름차순 정렬
        sorted_data = data.sort_values(by=prob_col).reset_index(drop=True)
        n = len(sorted_data)
        cut = int(n * proportion)
        
        # 상하위 10% 절사
        if n - 2 * cut > 0:
            trimmed_data = sorted_data.iloc[cut : n - cut]
        else:
            trimmed_data = sorted_data # 데이터가 적으면 절사 생략
            
        weights = trimmed_data[weight_col].values
        probs = trimmed_data[prob_col].values
        
        # 가중치 총합이 0인 경우 방어 로직
        if np.sum(weights) == 0:
            return np.mean(probs)
            
        # 가중 평균 연산: sum(확률 * 거리) / sum(거리)
        return np.average(probs, weights=weights)

    # 2. 5개 원시 감정에 대한 씬 대표 가중 확률 계산
    scene_vector = {}
    for col in prob_cols:
        scene_vector[col] = get_weighted_trimmed_mean(valid_df, col, weight_col='distance', proportion=0.1)
    
    # 3. 확률 벡터 정규화 (총합 1.0 맞춤)
    total_prob = sum(scene_vector.values())
    if total_prob > 0:
        scene_vector = {k: v / total_prob for k, v in scene_vector.items()}

    print("--- [영화 씬(Scene) 단위 대표 확률 벡터 도출] ---")
    for k, v in scene_vector.items():
        print(f"{k:12}: {v:.4f}")
        
    representative_emotion = max(scene_vector, key=scene_vector.get)
    print("\n" + "="*50)
    print(f"가중치 기반 1차 추론 감정: [{representative_emotion.split('_')[0]}]")
    print("이 5개의 수치를 EmotionAnchorSystem(엑셀 엔진)에 1회 통과시키면 최종 복합 감정 라벨이 도출됩니다.")
    print("="*50)
    
    return scene_vector

if __name__ == "__main__":
    # 특정 구간의 프레임들이 모두 합쳐진 merged CSV를 투입
    calculate_scene_representative_emotion('merged_crowd_emotions.csv')