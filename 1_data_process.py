import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. 데이터 로드
try:
    df = pd.read_csv('train.csv')
    print("성공: 원본 데이터를 불러왔습니다.")
except FileNotFoundError:
    print("에러: train.csv 파일을 찾을 수 없습니다.")
    exit()

# [컬럼명 자동 매핑] 오타 해결용
def find_col(possible_names, df_cols):
    for name in possible_names:
        if name in df_cols: return name
    return None

hr_col = find_col(['Heart_Rate', 'heart_rate', 'Heart Rate'], df.columns)

# 2. 신규 지표 생성 (Feature Engineering)

# [BMR] 기초대사량 계산 (Mifflin-St Jeor 공식)
def calculate_bmr(row):
    # 성별 인코딩 전 원본 값을 기준으로 계산
    is_male = str(row['Sex']).lower() in ['male', '0', 'm']
    bmr = (10 * row['Weight']) + (6.25 * row['Height']) - (5 * row['Age'])
    return bmr + 5 if is_male else bmr - 161

df['BMR'] = df.apply(calculate_bmr, axis=1)

# [Zone] 학습 데이터 라벨링 (심박수 데이터를 사용하여 Zone 생성)
# ※ 주의: Zone을 만든 후 심박수 데이터는 삭제합니다.
df['HR_max'] = 220 - df['Age']
df['HR_Ratio'] = df[hr_col] / df['HR_max']

def assign_zone(ratio):
    if ratio < 0.6: return 1
    elif ratio < 0.7: return 2
    elif ratio < 0.8: return 3
    elif ratio < 0.9: return 4
    else: return 5
df['Zone'] = df['HR_Ratio'].apply(assign_zone).astype('int8')

# 3. 데이터 정제 및 7차원 확정
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']).astype('int8') # M:1, F:0 (보통 알파벳 순)

# [핵심] 심박수, 체온, HR_max 등 센서 관련 컬럼 제외!
# 오직 '사용자 입력 가능 정보' + 'BMR' + 'Zone' 7개만 남깁니다.
final_features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Zone', 'BMR']
target = ['Calories']

train_df = df[final_features + target].copy()

# 4. 정규화 (StandardScaler) 적용

# 정규화 대상: 성별(범주형)과 칼로리(타겟)를 제외한 나머지 6개
scale_cols = ['Age', 'Height', 'Weight', 'Duration', 'Zone', 'BMR']

scaler = StandardScaler()
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols]).astype('float32')
train_df['Calories'] = train_df['Calories'].astype('float32')

# 5. 결과 저장
processed_path = 'model/train_data_scaled_7dim.csv'
train_df.to_csv(processed_path, index=False)

# [중요] 스케일러 저장 (나중에 7개 피처 순서대로 사용됨)
joblib.dump(scaler, 'model/scaler.pkl')

print("-" * 40)
print("✅ [7-Dimension] 전처리 완료!")
print(f"📍 사용된 피처: {final_features}")
print(f"📍 제외된 피처: Heart_Rate, Body_Temp, HR_max (센서리스 모델)")
print(f"📍 저장된 파일: {processed_path}")
print("-" * 40)
print(train_df.head(3))