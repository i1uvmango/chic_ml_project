# 운동 데이터 분석 및 칼로리 예측 모델 개발 보고서

## 1. 개요 (Project Overview)
본 프로젝트는 사용자의 신체 정보(성별, 나이, 키, 몸무게)와 운동 데이터(지속 시간, 심박수, 체온 등)를 활용하여 **소모 칼로리(Calories)**를 정밀하게 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다.


## 2. 데이터셋 명세 (Data Specification)
사용할 데이터의 주요 변수는 다음과 같으며, 모델 학습 효율성을 위해 적절한 타입 변환 및 스케일링을 적용합니다.

| 변수명 (Variable) | 설명 (Description) | 데이터 타입 | 전처리 방식 (Preprocessing) | 계산식 (Formula) |
| :--- | :--- | :--- | :--- | :--- |
| **Sex** | 성별 (Female: 0, Male: 1) | int8 | Binary (그대로 사용) | 사용자 입력 (개인 신체 정보) |
| **Age** | 나이 | float32 | Standard Scaler | 사용자 입력 (개인 신체 정보) |
| **Height** | 키 (cm) | float32 | Standard Scaler | 사용자 입력 (개인 신체 정보) |
| **Weight** | 몸무게 (kg) | float32 | Standard Scaler | 사용자 입력 (개인 신체 정보) |
| **Duration** | 운동 지속 시간 (min) | float32 | Standard Scaler | 사용자 입력 |
| **Heart_Rate** | 심박수 (bpm) | float32 | Standard Scaler | 사용자 입력 (운동 강도 지표) |
| **Body_Temp** | 체온 (°C) | float32 | Standard Scaler | 사용자 입력 (운동 강도 지표) |
| **Calories** | **Target** (소모 칼로리) | float32 | 비고: 예측 대상 | [Zone]에서 소모된 칼로리 + BMR|
| **HR_max** | 최대 심박수 (Derived) | float32 | Standard Scaler | `220 - Age` |
| **Zone** | 운동 강도 구간 (1~5) | float32 | Standard Scaler | $Score = HR_{ratio} \times (1 + \alpha \cdot \Delta Temp)$ ($\alpha=0.1$) |
| **BMR** | 기초대사량 (Derived) | float32 | Standard Scaler | Mifflin-St Jeor 식 |
| **BigMac_Count** | 섭취 빅맥 개수 | int | - | 사용자 입력 (식사) |
| **Daily_In** | 일일 섭취 칼로리 | float32 | - | `BigMac_Count * 550` |
| **Daily_Out** | 일일 총 소모 칼로리 | float32 | - | `(Zone * Hours) + BMR` |
| **Calorie_Deficit**| 일일 칼로리 결손량 | float32 | - | `Daily_Out - Daily_In` |
| **Days_to_Goal** | 5kg 감량 소요 기간 | float32 | - | `38,500 / Calorie_Deficit` |

### 2.1 사용자 입력 변수
* 성별
* 나이
* 키
* 몸무게
* 운동 지속 시간
* zone(운동강도)
* 시간
* BigMac 지수

### 빅맥 지수
* 기존엔 주식의 가격을 평가하는데 사용했으나, 본 프로젝트에서는 사용자의 1일 칼로리 섭취량을 휴리스틱하게 입력 받는다.
* 입력 받은 빅맥 지수는 일일 섭취 칼로리로 측정 되고, 기초대사량과 계산되어 5kg를 빼는데 필요한 날짜 계산에 사용된다.



## 3. 개발 계획 (Development Plan)

### 3.1. 데이터 전처리 (Data Preprocessing)
- **결측치 및 이상치 처리**: 데이터의 유효성 검사 및 정제.
- **파생 변수 생성**: `HR_max` (220-Age), `BMR` (Mifflin-St Jeor 식 등 활용) 등 도메인 지식을 활용한 변수 추가 확보(이미 데이터셋에 포함된 경우 유효성 검증).
- **스케일링**: 수치형 데이터에 대해 `StandardScaler`를 적용하여 모델의 수렴 속도 및 성능 향상 도모.

### 3.2. 모델링 (Modeling Strategy)
- **Baseline 모델**: Linear Regression 또는 간단한 MLP로 기준점 설정.
- **후보 알고리즘**:
    - XGBoost / LightGBM (Tabular 데이터에 강점)
    - MLP (Multi-Layer Perceptron) (비선형 관계 포착)
- **학습 및 검증**:
    - Train/Test Split (e.g., 8:2)
    - Cross-Validation을 통한 과적합 방지.

### 3.3. 평가 지표 (Evaluation Metrics)
- **MAE (Mean Absolute Error)**: 실제 칼로리와의 절대적인 오차 평균.
- **RMSE (Root Mean Squared Error)**: 큰 오차에 가중치를 두어 평가.
- **R² Score**: 모델의 설명력 검증.

## 4. 기대 효과 (Expected Outcomes)
- 개인별 신체 특성을 반영한 정밀한 칼로리 소모량 예측.
- 운동 강도 설정 및 건강 관리를 위한 정량적 지표 제공.