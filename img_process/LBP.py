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

# Original LBP
def get_lbp(image):
    height, width = image.shape
    lbp = np.zeros((height-2, width-2), dtype=np.uint8)
    for i in tqdm(range(1, height-1)):
        for j in range(1, width-1):
            center = image[i, j]
            code = 0
            code |= (image[i-1, j-1] > center) << 7
            code |= (image[i-1, j] > center) << 6
            code |= (image[i-1, j+1] > center) << 5
            code |= (image[i, j+1] > center) << 4
            code |= (image[i+1, j+1] > center) << 3
            code |= (image[i+1, j] > center) << 2
            code |= (image[i+1, j-1] > center) << 1
            code |= (image[i, j-1] > center) << 0
            lbp[i-1, j-1] = code
    return lbp

# Uniform LBP
def get_uniform_lbp(image):
    height, width = image.shape
    uniform_lbp = np.zeros((height-2, width-2), dtype=np.uint8)
    for i in tqdm(range(1, height-1)):
        for j in range(1, width-1):
            center = image[i, j]
            code = 0
            code |= (image[i-1, j-1] > center) << 7
            code |= (image[i-1, j] > center) << 6
            code |= (image[i-1, j+1] > center) << 5
            code |= (image[i, j+1] > center) << 4
            code |= (image[i+1, j+1] > center) << 3
            code |= (image[i+1, j] > center) << 2
            code |= (image[i+1, j-1] > center) << 1
            code |= (image[i, j-1] > center) << 0
            if (code & (code << 1) & 0xfe) != 0:
                uniform_lbp[i-1, j-1] = 0
            else:
                uniform_lbp[i-1, j-1] = code
    return uniform_lbp


# Completed Local Binary Count
def get_clbc(image):
    height, width = image.shape
    clbc = np.zeros((height-2, width-2), dtype=np.uint8)
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i, j]
            code = 0
            code += abs(image[i-1, j-1] - center)
            code += abs(image[i-1, j] - center)
            code += abs(image[i-1, j+1] - center)
            code += abs(image[i, j+1] - center)
            code += abs(image[i+1, j+1] - center)
            code += abs(image[i+1, j] - center)
            code += abs(image[i+1, j-1] - center)
            code += abs(image[i, j-1] - center)
            clbc[i-1, j-1] = code
    return clbc


# 이미지 불러오기
image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
image2 = image[1:-1, 1:-1]

# LBP 특징 추출 수행
time_start = time.time()
# lbp_image = get_lbp(image) # 4.4s - vivid structure
lbp_image = get_uniform_lbp(image) # 4.5s - barely visible structure
# lbp_image = get_clbc(image) # 3.0s - invisible structure
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

