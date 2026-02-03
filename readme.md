| 변수명 | 데이터 타입 | 설명 | 스케일링 여부 | 중요도/비고 |
| :--- | :--- | :--- | :--- | :--- |
| Sex | int8 | 성별 (Female: 0, Male: 1) | 미적용 | 이진(Binary) 분류 변수 |
| Age | float32 | 사용자 나이 | Standard | 대사량 및 최대 심박수 결정 요인 |
| Height | float32 | 사용자 키 (cm) | Standard | 신체 구성 및 BMR 계산의 기초 |
| Weight | float32 | 사용자 몸무게 (kg) | Standard | 운동 부하 및 에너지 소모량과 직결 |
| Duration | float32 | 운동 지속 시간 (min) | Standard | 총 소모 칼로리에 선형적 영향 |
| Heart_Rate | float32 | 운동 중 심박수 (bpm) | Standard | 실제 신체 부하를 나타내는 지표 |
| Body_Temp | float32 | 운동 중 체온 (°C) | Standard | 에너지 대사 및 과부하 감지 지표 |
| Calories | float32 | 소모 칼로리 (kcal) | 미적용 | Target(정답값) |
| HR_max | float32 | 최대 심박수 (220−Age) | Standard | 상대 강도 산출을 위한 기준점 |
| Zone | float32 | 주관적 운동 강도 (1~5) | Standard | 대화 테스트(Talk Test) 기반 지표 |
| BMR | float32 | 기초대사량 (kcal/day) | Standard | 개인별 에너지 소비의 Baseline |
