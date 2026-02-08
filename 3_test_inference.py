import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# 1. 모델 구조 정의 (학습 때와 동일)
# 1. 모델 구조 정의 (학습 때와 동일)
class CaloriePredictor(nn.Module):
    def __init__(self, input_dim):
        super(CaloriePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

# 2. 필수 파일 로드
try:
    scaler = joblib.load('model/scaler.pkl')
    model_state = torch.load('model/calorie_model_7dim.pth')
    model = CaloriePredictor(input_dim=7) # 7차원으로 변경
    model.load_state_dict(model_state)
    model.eval()
    print("✅ 모델 및 스케일러 로드 완료!")
except FileNotFoundError:
    print("에러: 모델(.pth) 또는 스케일러(.pkl) 파일이 없습니다. 먼저 학습을 완료하세요.")
    exit()

# --- [입력 검증 함수 정의] ---
def get_valid_input(prompt, input_type=float, condition=None, error_msg="잘못된 입력입니다."):
    while True:
        try:
            val = input(prompt).strip()
            if not val: # 빈 입력 처리
                print("⚠️ 필수 입력 사항입니다.")
                continue
            
            converted_val = input_type(val)
            
            if condition and not condition(converted_val):
                print(f"⚠️ {error_msg}")
                continue
            
            return converted_val
        except ValueError:
            print("⚠️ 숫자 형식으로 입력해주세요.")

# 3. 사용자 입력 섹션 (검증 루프 적용)
print("\n" + "="*40)
print("      DIET D-DAY PREDICTOR (AI)")
print("="*40)

# 성별 검증
while True:
    sex = input("1. 성별 (M/F): ").strip().upper()
    if sex in ['M', 'F']: break
    print("⚠️ M 또는 F만 입력 가능합니다.")

# 나머지 수치 검증
age = get_valid_input("2. 나이: ", float, lambda x: x > 0, "나이는 0보다 커야 합니다.")
height = get_valid_input("3. 키 (cm): ", float, lambda x: x > 0, "키는 0보다 커야 합니다.")
weight = get_valid_input("4. 몸무게 (kg): ", float, lambda x: x > 0, "몸무게는 0보다 커야 합니다.")
zone = int(get_valid_input("5. 운동 강도 (Zone 1~5 선택): ", float, lambda x: 1 <= x <= 5, "1에서 5 사이의 숫자를 선택하세요."))
workout_hours = get_valid_input("6. 하루 평균 운동 시간 (시간): ", float, lambda x: x >= 0, "시간은 0 이상이어야 합니다.")
bigmac_count = int(get_valid_input("7. 하루 빅맥 섭취 개수: ", float, lambda x: x >= 0, "개수는 0 이상이어야 합니다."))

# 4. 인퍼런스를 위한 내부 변수 생성 (Scientific Logic)
is_male = 1 if sex == 'M' else 0
bmr = (10 * weight) + (6.25 * height) - (5 * age) + (5 if is_male else -161)

# 심박수는 Zone 계산용으로만 사용하거나 로직 유지 (모델 입력에는 제외)
hr_max = 220 - age

# 5. AI 모델 예측 (7차원 입력 구성)
# 학습 데이터 순서: Sex, Age, Height, Weight, Duration, Zone, BMR
input_data = pd.DataFrame([[
    is_male, age, height, weight, 60.0, zone, bmr
]], columns=['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Zone', 'BMR'])

# 스케일링 대상: Sex와 Target을 제외한 나머지
scale_cols = ['Age', 'Height', 'Weight', 'Duration', 'Zone', 'BMR']
input_scaled = input_data.copy()
input_scaled[scale_cols] = scaler.transform(input_data[scale_cols])

# 6. 최종 다이어트 계산
with torch.no_grad():
    pred_cal_per_hour = model(torch.tensor(input_scaled.values, dtype=torch.float32)).item()

daily_out = (pred_cal_per_hour * workout_hours) + bmr
daily_in = bigmac_count * 550
daily_deficit = daily_out - daily_in

print("\n" + "-"*40)
print(f"📊 나의 기초대사량(BMR): {bmr:.1f} kcal")
print(f"🔥 AI 예측 운동 소모량(시간당): {pred_cal_per_hour:.1f} kcal")
print(f"⚖️ 일일 칼로리 결손량: {daily_deficit:.1f} kcal")

if daily_deficit <= 0:
    print("⚠️ 경고: 섭취량이 더 많습니다! 현재 생활로는 감량이 불가능합니다.")
else:
    target_kcal = 38500  # 5kg 감량 목표
    days = target_kcal / daily_deficit
    print(f"📅 5kg 감량까지 예상 소요 기간: {int(days)}일")
print("-"*40)