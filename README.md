# Credit Score Classification

신용카드 사용자의 월별 금융 데이터를 기반으로 신용등급(Poor / Standard / Good)을 분류하는 딥러닝 모델 개발

## 데이터

- 원본 출처: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- 데이터 클리닝 출처: [Credit Score Classification Part 1 - Data Cleaning](https://www.kaggle.com/code/clkmuhammed/credit-score-classification-part-1-data-cleaning)
- 사용 데이터: 위 클리닝 노트북 기준으로 1차 전처리된 `train2.csv`
- 구조: 12,500명 × 8개월 = 100,000행, 28개 피처
- 타겟: Credit_Score (Poor / Standard / Good)

## 결과

| Metric | Score |
|--------|-------|
| Validation Accuracy | 0.XXXX |

![Confusion Matrix](images/confusion_matrix.png)
![클래스 분포](images/pred_distribution.png)

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| [전처리](docs/01_preprocessing.md) | 데이터 탐색, 인코딩, 멀티레이블 분해 |
| [EDA](docs/02_eda.md) | 수치형/범주형 분포, 상관관계, 핵심 변수 확인 |
| [변수선정](docs/03_feature_selection.md) | 제거 컬럼, 파생변수 5개 생성 |
| [모델링](docs/04_model.md) | MLP 베이스라인 → TabTransformer 개선, 한계점 |

---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
