import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 1. 모델 구조 (기존과 동일)
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

# 2. 모델 로드 및 가중치 추출
model = CaloriePredictor(input_dim=10)
model.load_state_dict(torch.load('calorie_model.pth'))
model.eval()

# 첫 번째 레이어(10 -> 128)의 가중치 평균값 계산
# 각 입력 변수가 128개의 노드에 미치는 영향력의 평균을 구합니다.
first_layer_weights = model.net[0].weight.data.numpy()
feature_importance = np.mean(np.abs(first_layer_weights), axis=0)

# 3. 결과 출력
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'HR_max', 'Zone', 'BMR']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("-" * 30)
print("📊 모델 체크포인트 역산 결과 (변수 중요도)")
print("-" * 30)
print(importance_df)