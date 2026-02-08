import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

# 1. 데이터 로드 (7차원 전용 데이터)
file_path = 'model/train_data_scaled_7dim.csv'
if not os.path.exists(file_path):
    print(f"에러: {file_path} 파일이 없습니다. 전처리 코드를 먼저 실행하세요.")
    exit()

df = pd.read_csv(file_path)

# 입력 변수(X)와 정답값(y) 분리
# X: Sex, Age, Height, Weight, Duration, Zone, BMR (총 7개)
X = df.drop(columns=['Calories']).values
y = df['Calories'].values.reshape(-1, 1)

# 학습 데이터와 검증 데이터 분리 (8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# DataLoader 설정
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

# 2. 7차원 전용 MLP 모델 구조
class CaloriePredictor(nn.Module):
    def __init__(self, input_dim=7): # 입력층 구멍을 7개로 고정!
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

    def forward(self, x):
        return self.net(x)

# 모델 초기화
model = CaloriePredictor(input_dim=7) # 명시적으로 7 지정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 모델 학습 (Training Loop)
epochs = 20
print(f"🚀 7차원 센서리스 모델 학습 시작 (Feature: {X.shape[1]}개)")



for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증 (Validation)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_output = model(batch_X)
            v_loss = criterion(val_output, batch_y)
            val_loss += v_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# 4. 모델 저장 (7차원 전용 파일명)
torch.save(model.state_dict(), 'model/calorie_model_7dim.pth')
print("-" * 30)
print("✅ 학습 완료! 모델 저장: model/calorie_model_7dim.pth")