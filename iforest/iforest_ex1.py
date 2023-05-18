#!/usr/bin/env python
# coding: utf8

###########################
####### iForest test ######
###########################
# 참고
#   - https://partrita.github.io/posts/isolation-forest/
#   - https://john-analyst.medium.com/isolation-forest%EB%A5%BC-%ED%86%B5%ED%95%9C-%EC%9D%B4%EC%83%81%ED%83%90%EC%A7%80-%EB%AA%A8%EB%8D%B8-9b10b43eb4ac
#   - 우리 data는 label 없고, feature가 많다.(따라서 이거 쓰기에 적합하다 판단)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline  # notebook을 실행한 브라우저에서 바로 그림을 볼 수 있게 해주는 옵션
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

# hyper parameter 세팅
seed = 42           # random seed
max_samples = 100   # base estimator를 만드는데 필요한 data 개수
max_features = 1    # base estimator를 만드는데 필요한 feature 개수
outlier_rate = 0.05 # outlier 비율  (기본 : 0.1)
PCA_dim = 3         # 결과 확인할때 쓰는 차원축소 할 차원수

subplot_fig_size = 6   # 큰 figure 안에 subplot

# 간단한 data 생성
rng = np.random.RandomState(seed)
# Generating training data 
X_train = 0.2 * rng.randn(1000, 2)
#X_train = np.r_[X_train + 3, X_train]  # -> 아래와 같이 다르게 표현
X_train = np.concatenate((X_train+3, X_train), axis = 0)
X_train_df = pd.DataFrame(X_train, columns = ["x1", "x2"])
# Generating new, "normal" observation
X_test = 0.2 * rng.randn(200, 2)
#X_test = np.r_[X_test + 3, X_test]  # -> 아래와 같이 다르게 표현
X_test = np.concatenate((X_test+3, X_test), axis = 0)
X_test_df = pd.DataFrame(X_test, columns = ["x1", "x2"])
# Generating outliers
X_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
X_outliers_df = pd.DataFrame(X_outliers, columns = ["x1", "x2"])



# scikit-learn으로 학습 모델 만들기
clf = IsolationForest(max_samples=max_samples, 
                        max_features=max_features,
                        contamination = outlier_rate, 
                        random_state=seed)


#####################################################
# 추가 테스트: train과 test를 각각 다른 outlier와 함께 병합
if seed <= 0:
    print("[ERROR] select seed value higher than 1")
    assert(False)
rng2 = np.random.RandomState(seed+1)
X_outliers_merge = rng2.uniform(low=-1, high=5, size=(100, 2))
X_outliers_merge = pd.DataFrame(X_outliers_merge, columns = ["x1", "x2"])
X_train_df = pd.concat([X_train_df, X_outliers_merge], axis=0)

rng3 = np.random.RandomState(seed+2)
X_outliers_merge2 = rng3.uniform(low=-1, high=5, size=(100, 2))
X_outliers_merge2 = pd.DataFrame(X_outliers_merge2, columns = ["x1", "x2"])
X_test_df = pd.concat([X_test_df, X_outliers_merge2], axis=0)

#####################################################
# 정답 세팅
y_true_train = np.concatenate((np.ones(X_train.shape[0]), (-1) * np.ones(X_outliers_merge.shape[0])), axis=0)
y_true_test = np.concatenate((np.ones(X_test.shape[0]), (-1) * np.ones(X_outliers_merge2.shape[0])), axis=0)
y_true_outlier = (-1) * np.ones(X_outliers.shape[0])

#####################################################
# fitting + prediction
clf.fit(X_train_df)
y_pred_train = clf.predict(X_train_df)
y_pred_test = clf.predict(X_test_df)
y_pred_outliers = clf.predict(X_outliers_df)

########################################################
# 한번에 시각화
fig = plt.figure(figsize=[ 4*subplot_fig_size, subplot_fig_size ])
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)

# origin data
ax1.scatter(X_train_df.x1, X_train_df.x2, c="white", s=100, edgecolor='k', label="train data")
#ax1.scatter(X_test_df.x1, X_test_df.x2, c="green", s=50, edgecolor='k', label="test data")
#ax1.scatter(X_outliers_df.x1, X_outliers_df.x2, c="red", s=50, edgecolor='k', label="outliers")
ax1.set_title("origin")
ax1.set_xlim([-1,5])
ax1.set_ylim([-1,5])
ax1.legend(loc="upper right")


# outlier pred 결과
X_outliers_df = X_outliers_df.assign(y = y_pred_outliers)
ax2.scatter(X_train_df.x1, X_train_df.x2, c="white",
                 s=50, edgecolor='k', label="train data")
ax2.scatter(X_outliers_df.loc[X_outliers_df.y == -1, ["x1"]], 
                 X_outliers_df.loc[X_outliers_df.y == -1, ["x2"]], 
                 c="red", s=50, edgecolor='k', label="correct outlier prediction")
#ax2.scatter(X_outliers_df.loc[X_outliers_df.y == 1, ["x1"]], 
#                 X_outliers_df.loc[X_outliers_df.y == 1, ["x2"]], 
#                 c="green", s=50, edgecolor='k', label="incorrect outlier prediction")
ax2.set_title("train prediction")
ax2.set_xlim([-1,5])
ax2.set_ylim([-1,5])
ax2.legend(loc="upper right")


# outlier pred 결과
X_test_df = X_test_df.assign(y = y_pred_test)
ax3.scatter(X_train_df.x1, X_train_df.x2, c="white",
                 s=50, edgecolor='k', label="train data")
ax3.scatter(X_test_df.loc[X_test_df.y == -1, ["x1"]], 
                 X_test_df.loc[X_test_df.y == -1, ["x2"]], 
                 c="red", s=50, edgecolor='k', label="correct outlier prediction")
#ax3.scatter(X_test_df.loc[X_test_df.y == 1, ["x1"]], 
#                 X_test_df.loc[X_test_df.y == 1, ["x2"]], 
#                 c="green", s=50, edgecolor='k', label="incorrect outlier prediction")
ax3.set_title("test prediction")
ax3.set_xlim([-1,5])
ax3.set_ylim([-1,5])
ax3.legend(loc="upper right")
########################################################



# 모델 정확도
print("학습 데이터셋에서 정확도:", list(y_pred_train).count(1)/y_pred_train.shape[0])
print("학습 데이터셋에서 F1 score: {}\n".format(f1_score(y_true_train, y_pred_train)))

print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1)/y_pred_test.shape[0])
print("테스트 데이터셋에서 F1 score: {}\n".format(f1_score(y_true_test, y_pred_test)))

print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
print("이상치 데이터셋에서 F1 score: {}\n".format(f1_score(y_true_outlier, y_pred_outliers)))



#######################################################
# testset 데이터를 PCA로 차원 축소 시켜서 잘 나눠졌는지 확인
# 2개 차원으로 축소
if 2 == PCA_dim:
    pca = PCA(PCA_dim)
    pca.fit(X_test_df)
    res = pd.DataFrame(pca.transform(X_test_df))   # PCA 주성분으로 data 변환
    Z = np.array(res)

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title("PCA to 2-dimension")
    ax4.scatter(res[0], res[1], c="green", s=20, label="normal points")
    outliers = res.loc[y_pred_test == -1]   # outlier로 prediction 된 곳들만 골라서
    outlier_index = list(outliers.index)    # index 가져오기
    ax4.scatter(res.iloc[outlier_index, 0], # PCA 차원축소 후 column은 1, 2, ... 로 변함. 따라서 iloc 사용
                res.iloc[outlier_index, 1],
                c="green", s=20, edgecolor="red", label="predicted outliers")
    ax4.legend(loc="upper right")

# 3개 차원으로 축소
elif 3 == PCA_dim:
    from sklearn.preprocessing import StandardScaler
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=PCA_dim)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_test_df)
    X_reduce = pca.fit_transform(X)
    
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    #ax4.set_zlabel("x_composite_3")
    ax4.scatter(X_reduce[:, 0], X_reduce[:, 1], X_reduce[:, 2], 
                s=10, label="inliers", c="green")
    ax4.legend("upper right")

else:
    print("[ERROR] check PCA_dim")
    assert(False)
plt.show()
d=1