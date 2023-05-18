#! usr/bin/env python
#-*- encoding: utf-8 -*-

import xgboost
import shap

# 예시 데이터셋 및 모델 학습
X, y = shap.datasets.diabetes()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# Explainer 객체와 SHAP 값을 생성합니다.
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 설정한 감쇠 파라미터 (phi)에 따른 Weighted SHAP 값을 계산합니다. (예: phi = 0.1)
phi = 0.5
weighted_shap_values = explainer(X, phi)

# 원래의 SHAP 값과 Weighted SHAP 값을 동시에 플롯하기
shap.summary_plot(shap_values, X)
shap.summary_plot(weighted_shap_values, X)
print()