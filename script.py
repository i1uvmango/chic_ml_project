import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("에러: train.csv 파일을 찾을 수 없습니다.")
    exit()

# 컬럼명 자동 매핑
def find_col(possible_names, df_cols):
    for name in possible_names:
        if name in df_cols: return name
    return None

hr_col = find_col(['Heart_Rate', 'heart_rate', 'Heart Rate'], df.columns)
temp_col = find_col(['Body_Temp', 'Body_Tem', 'body_temp', 'Body Temperature'], df.columns)

# 2. 신규 지표 계산 (HR_max, Zone, BMR)

# [HR_max] 최대 심박수
df['HR_max'] = 220 - df['Age']

# [HR_Ratio] 강도 계산의 핵심 지표
df['HR_Ratio'] = df[hr_col] / df['HR_max']

# [Zone] 대화 테스트(RPE) 기준에 맞춘 고정 임계값 분류
# 스포츠 과학 표준 가이드를 따름
def assign_zone(ratio):
    if ratio < 0.6: return 1
    elif ratio < 0.7: return 2
    elif ratio < 0.8: return 3
    elif ratio < 0.9: return 4
    else: return 5

df['Zone'] = df['HR_Ratio'].apply(assign_zone).astype('int8')

# [BMR] 기초대사량 (Mifflin-St Jeor)
def calculate_bmr(row):
    is_male = str(row['Sex']).lower() in ['male', '0', 'm']
    bmr = (10 * row['Weight']) + (6.25 * row['Height']) - (5 * row['Age'])
    return bmr + 5 if is_male else bmr - 161

df['BMR'] = df.apply(calculate_bmr, axis=1)

# 3. 데이터 정제 및 최종 컬럼 구성 (11개)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']).astype('int8')

# 요청하신 11개 컬럼 순서 고정
final_cols = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 
    hr_col, temp_col, 'Calories', 'HR_max', 'Zone', 'BMR'
]

train_df = df[final_cols].copy()
train_df.columns = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 
    'Heart_Rate', 'Body_Temp', 'Calories', 'HR_max', 'Zone', 'BMR'
]

# 4. 결과 저장
processed_path = 'train_data.csv'
train_df.to_csv(processed_path, index=False)

print("-" * 30)
print("전처리 완료 (대화 지수 기반 Zone 적용)!")
print(f"총 데이터 수: {len(train_df):,}개")
print(f"컬럼 구성: {train_df.columns.tolist()}")
print("-" * 30)
print(train_df[['Heart_Rate', 'HR_max', 'Zone']].head(5)) # Zone 배정 확인용