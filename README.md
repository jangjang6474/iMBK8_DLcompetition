# Credit Score Classification

## 프로젝트 개요
신용카드 사용자의 월별 금융 데이터를 기반으로 신용등급(Poor / Standard / Good)을 분류하는 딥러닝 모델 개발

## 데이터
- 출처: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- 구조: 12,500명 × 8개월 = 100,000행, 28개 피처
- 타겟: Credit_Score (Poor / Standard / Good)

## 전처리
- 식별자 컬럼 제거 (ID, Name, SSN, Customer_ID)
- Type_of_Loan 멀티레이블 → 더미변수 8개로 분해
- Payment_Behaviour → spent_level, payment_size 2개 컬럼으로 분리
- Credit_Mix, Payment_of_Min_Amount Ordinal 인코딩
- Occupation Label 인코딩

## EDA
- 수치형: 박스플롯, 히스토그램, 클래스별 분포 비교
- 범주형: 교차분석, 직업별 신용점수 히트맵, 대출종류별 분포
- Credit_Mix, Interest_Rate, Credit_History_Age가 핵심 예측 변수로 확인

![EDA](images/eda.png)

## Feature Selection
- 타겟 상관 거의 없는 Total_EMI_per_month, Credit_Utilization_Ratio 제거
- 파생변수 5개 생성
  - debt_to_income: 부채/소득 비율
  - delay_severity: 연체 횟수 × 연체 기간
  - inquiry_per_card: 카드 수 대비 신용 조회수
  - balance_to_salary: 월수입 대비 잔액
  - loan_burden: 대출 수 × 이자율

## 모델
- **TabTransformer**
  - dim=32, depth=3, heads=4
  - MLP: GeLU 활성화, 항아리 형태 (dim×4 → dim×2 → 3)
  - Dropout: 0.1
- 옵티마이저: NAdam (lr=1e-3, weight_decay=1e-5)
- 스케줄러: StepLR (step_size=5, gamma=0.5)
- 분리: stratify 기반 8:2 무작위 분할

## 결과
| Metric | Score |
|--------|-------|
| Validation Accuracy | 0.XXXX |

![Confusion Matrix](images/confusion_matrix.png)
![클래스 분포](images/pred_distribution.png)
