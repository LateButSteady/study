#!usr/bin/env python
#-*- encoding: utf-8 -*-

import os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dir_now = os.path.dirname(os.path.abspath(__file__))
fname_img = 'train-volume.tif'
path_img = os.path.join(dir_now, fname_img)

# Original HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 이미지 불러오기
image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

# HOG 특징 추출 수행
locations, _ = hog.detectMultiScale(image)
cv2.imshow('HOG Result', image)
cv2.waitKey()
cv2.destroyAllWindows()

# HOG 결과 시각화
for (x, y, w, h) in locations:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# LBP 특징 추출 수행
time_start = time.time()
print("[INFO] Elapsed time : ", time.time() - time_start)

# LBP 이미지 출력
plt.figure(1)
plt.subplot(3,1,1)
plt.imshow(image2[200:230, 200:230])
plt.subplot(3,1,2)
plt.imshow(lbp_image[200:230, 200:230])
plt.subplot(3,1,3)
plt.imshow(image2[200:230, 200:230] - lbp_image[200:230, 200:230])
plt.show()
# cv2.imshow('LBP Image', lbp_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

