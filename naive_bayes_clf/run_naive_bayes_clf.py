#! usr/bin/env python
#-*- encoding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

dict_feature = {
        0: "Height", 
        1: "Weight", 
        2: "Foot size", 
        3: "Muscle Percent",}
ind_feature = 3

# Train data
train_data = np.array([
    [1.75, 65, 10, 30, 0],
    [1.80, 70, 11, 32, 0],
    [1.82, 72, 12, 31, 0],
    [1.76, 68, 11, 29, 0],
    [1.88, 80, 13, 32, 0],

    [1.78, 75, 11, 30, 0],
    [1.82, 80, 12, 32, 0],
    [1.85, 85, 13, 33, 0],
    [1.82, 72, 11, 31, 0],
    [1.76, 68, 10, 30, 0],

    [1.88, 80, 12, 32, 0],
    [1.90, 85, 13, 33, 0],
    [1.62, 55, 8, 26, 1],
    [1.65, 58, 9, 27, 1],
    [1.68, 62, 10, 28, 1],

    [1.62, 55, 8, 26, 1],
    [1.70, 60, 9, 27, 1],
    [1.64, 58, 8, 28, 1],
    [1.68, 62, 9, 28, 1],
    [1.70, 66, 10, 29, 1],
])

# Test data
test_data = np.array([
    [1.62, 68, 11, 31],
    [1.60, 55, 8, 27],
    [1.56, 70, 10, 30],
    [1.68, 75, 12, 32],
    [1.73, 58, 8, 27],

    [1.68, 62, 9, 29],
    [1.75, 65, 10, 30],
    [1.80, 70, 11, 32],
    [1.68, 62, 9, 28],
    [1.60, 63, 9, 29],
])


def main():

    # 클래스별 사전 확률 계산
    num_male = int(train_data.shape[0] * 0.6)
    num_female = train_data.shape[0] - num_male
    p_male = num_male / train_data.shape[0]
    p_female = num_female / train_data.shape[0]

    # 클래스별 특성의 평균과 표준편차 계산
    male_data = train_data[train_data[:, -1] == 0][:, :-1]
    female_data = train_data[train_data[:, -1] == 1][:, :-1]
    male_means = np.mean(male_data, axis=0)
    male_stds = np.std(male_data, axis=0)
    female_means = np.mean(female_data, axis=0)
    female_stds = np.std(female_data, axis=0)

    # 테스트 데이터가 남자 분포에서 온지 여부 판별
    is_male_distribution = np.prod(norm.pdf(test_data[:, 0], male_means[0], male_stds[0])) * \
                        np.prod(norm.pdf(test_data[:, 1], male_means[1], male_stds[1])) * \
                        np.prod(norm.pdf(test_data[:, 2], male_means[2], male_stds[2])) * \
                        np.prod(norm.pdf(test_data[:, 3], male_means[3], male_stds[3]))

    is_female_distribution = np.prod(norm.pdf(test_data[:, 0], female_means[0], female_stds[0])) * \
                        np.prod(norm.pdf(test_data[:, 1], female_means[1], female_stds[1])) * \
                        np.prod(norm.pdf(test_data[:, 2], female_means[2], female_stds[2])) * \
                        np.prod(norm.pdf(test_data[:, 3], female_means[3], female_stds[3]))

    # 분류 결과 출력
    print("Likelyhood of male  : ", is_male_distribution)
    print("Likelyhood of female: ", is_female_distribution)
    if is_male_distribution >= is_female_distribution:
        print("Test data is from male distribution.")
    else:
        print("Test data is from female distribution.")

    plot_dist_and_test_data(ind_feature)



def plot_dist_and_test_data(ind_feature):
    
    # 클래스별 데이터 추출
    male_data = train_data[train_data[:, -1] == 0][:, ind_feature]
    female_data = train_data[train_data[:, -1] == 1][:, ind_feature]

    # 클래스별 평균과 표준편차 계산
    male_mean = np.mean(male_data)
    male_std = np.std(male_data)
    female_mean = np.mean(female_data)
    female_std = np.std(female_data)

    # Figure 생성
    fig = plt.figure()

    # 남자 분포 플롯
    x_male = np.linspace(male_mean - 3 * male_std, male_mean + 3 * male_std, 100)
    y_male = 1 / (np.sqrt(2 * np.pi) * male_std) * np.exp(-0.5 * ((x_male - male_mean) / male_std) ** 2)
    plt.plot(x_male, y_male, color='blue', label='Male Distribution')
    # 여자 분포 플롯
    x_female = np.linspace(female_mean - 3 * female_std, female_mean + 3 * female_std, 100)
    y_female = 1 / (np.sqrt(2 * np.pi) * female_std) * np.exp(-0.5 * ((x_female - female_mean) / female_std) ** 2)
    plt.plot(x_female, y_female, color='red', label='Female Distribution')

    # 테스트 데이터 플롯
    plt.scatter(test_data[:, ind_feature], np.zeros_like(test_data[:, ind_feature]), color='green', label='Test Data')

    # 축과 제목 설정
    # plt.xlabel('Height')
    plt.ylabel('Probability Density')
    plt.title(dict_feature[ind_feature])

    # 범례 추가
    plt.legend()

    # 그래프 표시
    plt.show()



if __name__ == "__main__":
    main()