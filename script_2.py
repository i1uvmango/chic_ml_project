import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. 데이터 로드
try:
    # 75만 개 대용량 데이터 로드
    df = pd.read_csv('train.csv')
    print("성공: 원본 데이터를 불러왔습니다.")
except FileNotFoundError:
    print("에러: train.csv 파일을 찾을 수 없습니다.")
    exit()

# [컬럼명 자동 매핑] 'Body_Tem' 등 오타나 명칭 차이 해결
def find_col(possible_names, df_cols):
    for name in possible_names:
        if name in df_cols: return name
    return None

hr_col = find_col(['Heart_Rate', 'heart_rate', 'Heart Rate'], df.columns)
temp_col = find_col(['Body_Temp', 'Body_Tem', 'body_temp', 'Body Temperature'], df.columns)

# 2. 신규 지표 생성 (Feature Engineering)

# [HR_max] 최대 심박수 계산
df['HR_max'] = 220 - df['Age']

# [HR_Ratio] 강도 계산을 위한 상대적 심박수
df['HR_Ratio'] = df[hr_col] / df['HR_max']

# [Zone] 대화 테스트(Talk Test) 기준 기반 분류 (Scientific Threshold)
def assign_zone(ratio):
    if ratio < 0.6: return 1   # 노래 가능
    elif ratio < 0.7: return 2 # 편안한 대화 가능
    elif ratio < 0.8: return 3 # 짧은 문장 대화 가능
    elif ratio < 0.9: return 4 # 한두 단어 간신히 대화 가능
    else: return 5             # 대화 불가
df['Zone'] = df['HR_Ratio'].apply(assign_zone).astype('int8')

# [BMR] 기초대사량 계산 (Mifflin-St Jeor 공식)
def calculate_bmr(row):
    is_male = str(row['Sex']).lower() in ['male', '0', 'm']
    bmr = (10 * row['Weight']) + (6.25 * row['Height']) - (5 * row['Age'])
    return bmr + 5 if is_male else bmr - 161
df['BMR'] = df.apply(calculate_bmr, axis=1)

# 3. 데이터 정제 및 컬럼 정렬 (요청하신 11개 구성)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']).astype('int8')

# 최종 11개 컬럼 선택 및 이름 통일
final_cols = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 
    hr_col, temp_col, 'Calories', 'HR_max', 'Zone', 'BMR'
]
train_df = df[final_cols].copy()
train_df.columns = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration', 
    'Heart_Rate', 'Body_Temp', 'Calories', 'HR_max', 'Zone', 'BMR'
]

# 4. 정규화 (StandardScaler) 적용

# 정규화 대상: 연속형 변수들 (성별과 타겟인 칼로리 제외)
scale_cols = [
    'Age', 'Height', 'Weight', 'Duration', 
    'Heart_Rate', 'Body_Temp', 'HR_max', 'Zone', 'BMR'
]

scaler = StandardScaler()
# 75만 개 연산 효율을 위해 float32로 변환하며 스케일링
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols]).astype('float32')
train_df['Calories'] = train_df['Calories'].astype('float32')

# 5. 결과 저장 (CSV 및 Scaler PKL)
processed_path = 'train_data_scaled.csv'
train_df.to_csv(processed_path, index=False)

# [중요] 나중에 앱(Inference)에서 입력값 변환을 위해 스케일러 저장
joblib.dump(scaler, 'scaler.pkl')

print("-" * 40)
print("✅ 전처리 및 정규화 통합 완료!")
print(f"📍 총 데이터 수: {len(train_df):,}개")
print(f"📍 저장된 파일: {processed_path}")
print(f"📍 스케일러 저장: scaler.pkl")
print("-" * 40)
print("📊 정규화 후 데이터 상위 3행:")
print(train_df.head(3))