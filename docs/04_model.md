# 04. 모델링

## 베이스라인 모델 (MLP)

> [!NOTE]
> **모델 구조**
> - 3층 MLP: 64 → 32 → 3
> - 활성화 함수: ReLU
> - 옵티마이저: Adam (lr=0.001)
> - 에폭: 20, 배치: 32
> - 전처리: StandardScaler 전체 적용, Type_of_Loan 단순 Label Encoding

> [!WARNING]
> **한계점**
> - Type_of_Loan 멀티레이블을 단순 Label Encoding 적용 (조합 폭발 문제)
> - 범주형/수치형 구분 없이 전체 스케일링
> - 클래스 불균형 미처리
> - 단순 구조로 피처 간 복잡한 관계 학습 어려움

---

## 개선 모델 — TabTransformer

### 선택 이유
- 범주형 피처(13개)가 많은 이 데이터에서 트랜스포머 어텐션이 범주형 임베딩 간 관계를 효과적으로 학습
- 수치형은 MLP로, 범주형은 트랜스포머로 분리 처리하는 구조가 데이터 특성에 적합

### 모델 구조

```python
model = TabTransformer(
    categories=tuple(cat_dims),       # 범주형 카디널리티
    num_continuous=len(num_col_names), # 수치형 피처 수 (13)
    dim=32,                            # 트랜스포머 임베딩 차원
    depth=3,                           # 트랜스포머 레이어 수
    heads=4,                           # 멀티헤드 어텐션 수
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),           # 항아리 형태 MLP (dim×4 → dim×2 → output)
    mlp_act=nn.GELU(),                 # GeLU 활성화
    num_special_tokens=2,
    dim_out=3,                         # Poor / Standard / Good
)
```

### 학습 설정

| 항목 | 설정값 |
|------|-------|
| 옵티마이저 | NAdam (lr=1e-3, weight_decay=1e-5) |
| 스케줄러 | StepLR (step_size=5, gamma=0.5) |
| Loss | CrossEntropyLoss |
| 배치 크기 | 512 |
| 최대 에폭 | 100 |
| 조기 종료 | patience=10 |

### 데이터 분리

```python
# stratify 기반 8:2 무작위 분할
X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# 수치형만 StandardScaler 적용
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_valid = scaler.transform(X_num_valid)
```

> [!NOTE]
> TabTransformer는 내부 배치 정규화가 없어 수치형 스케일링이 필수이며,
> 범주형 컬럼은 임베딩으로 처리하므로 스케일링 대상에서 제외하였다.

---

## ver1 vs ver2 비교

| 항목 | ver1 | ver2 |
|------|------|------|
| 데이터 분리 | GroupShuffleSplit (고객 단위) | 행 단위 무작위 (stratify) |
| 클래스 가중치 | balanced 가중치 적용 | 일반 CEL |
| Validation Accuracy | 0.65 | **0.XXXX** |

> [!NOTE]
> ver1에서 고객 단위 분리 시 검증셋이 학습에 없던 고객만으로 구성되어
> 모델이 새로운 고객 패턴을 일반화하지 못하고 성능이 65%대에 머물렀다.
> ver2에서 행 단위 무작위 분리로 변경 후 성능이 개선되었다.

---

## 최종 결과

<img width="431" height="197" alt="image" src="https://github.com/user-attachments/assets/e18813a9-1130-46f9-b0ac-69a958f96fb7" />

![Confusion Matrix](../images/confusion_matrix.png)

![클래스 분포](../images/pred_distribution.png)

---

## 한계점 및 시사점

> [!WARNING]
> - 빅데이터 시대에 특정 개인의 누적 이력 없이도 동일 시점의 집계 데이터만으로 신용등급을 예측할 수 있음을 확인하였다.
> - 단, 학습 데이터가 특정 연도 1\~8월에 한정되므로 해당 기간 분포를 벗어난 시점의 일반화 성능은 보장하기 어렵다.
> - 큰 경제적 충격이 없는 안정적인 환경을 전제로, 동일한 월별 집계 구조의 데이터에 한해 동일 연도 9\~12월 신용등급 예측에도 유의미한 결과를 기대할 수 있다.
