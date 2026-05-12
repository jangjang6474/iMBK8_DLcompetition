# 03. Feature Selection

## 제거 컬럼

EDA 상관관계 분석 결과를 바탕으로 타겟과 상관이 거의 없는 컬럼을 제거하였다.

```python
data.drop(columns=['Total_EMI_per_month', 'Credit_Utilization_Ratio'], inplace=True)
```

| 컬럼 | 제거 이유 |
|------|----------|
| Total_EMI_per_month | 타겟 상관계수 0.02, 대출 수·소득과 중복 정보 |
| Credit_Utilization_Ratio | 타겟 상관계수 0.05, 예측력 거의 없음 |

---

## 파생변수 생성

EDA에서 강한 예측력을 보인 변수들의 조합으로 신용 위험 및 재정 상태를 압축한 지표를 생성하였다.

```python
# 부채 대비 소득 — 재정 압박 지표
data['debt_to_income'] = data['Outstanding_Debt'] / data['Annual_Income']

# 연체 심각도 — 연체 횟수 × 연체 기간
data['delay_severity'] = data['Delay_from_due_date'] * data['Num_of_Delayed_Payment']

# 카드 대비 신용 조회수 — 신용 과다 사용 신호
data['inquiry_per_card'] = data['Num_Credit_Inquiries'] / (data['Num_Credit_Card'] + 1)

# 월수입 대비 잔액 — 저축 여력 지표
data['balance_to_salary'] = data['Monthly_Balance'] / (data['Monthly_Inhand_Salary'] + 1)

# 대출 부담도 — 대출 수 × 이자율
data['loan_burden'] = data['Num_of_Loan'] * data['Interest_Rate']
```

| 파생변수 | 조합 | 의미 |
|---------|------|------|
| debt_to_income | Outstanding_Debt / Annual_Income | 재정 압박 수준 |
| delay_severity | Delay × Num_of_Delayed_Payment | 연체 심각도 |
| inquiry_per_card | Num_Credit_Inquiries / (Num_Credit_Card+1) | 신용 과다 사용 |
| balance_to_salary | Monthly_Balance / (Monthly_Inhand_Salary+1) | 저축 여력 |
| loan_burden | Num_of_Loan × Interest_Rate | 대출 부담도 |

---

## 다중공선성 방지를 위한 원본 컬럼 제거

파생변수 생성에 사용된 원본 컬럼을 제거하여 다중공선성을 방지하였다.

```python
drop_cols = [
    'Outstanding_Debt', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Num_Credit_Inquiries', 'Monthly_Balance', 'Num_of_Loan', 'Interest_Rate'
]
data.drop(columns=drop_cols, inplace=True)
```

---

## 최종 피처 구성

| 구분 | 피처 수 | 목록 |
|------|--------|------|
| 수치형 | 13개 | Age, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Changed_Credit_Limit, Credit_History_Age, Amount_invested_monthly, debt_to_income, delay_severity, inquiry_per_card, balance_to_salary, loan_burden |
| 범주형 | 13개 | Occupation, Credit_Mix, Payment_of_Min_Amount, spent_level, payment_size, loan_auto_loan × 8종 |
| **합계** | **26개** | |
