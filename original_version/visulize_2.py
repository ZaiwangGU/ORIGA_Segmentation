import cv2
import numpy as np
import matplotlib.pyplot as plt


origin_image = cv2.imread("E:\\battleNet10_1\data\different_threshold\AGLAIA_GT_521.jpg")
gray_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2GRAY)
print(gray_image)
# gray = cv2.cvtColor(origin_image, cv2.)
ret, binary = cv2.threshold(gray_image, 0.5, 1, cv2.THRESH_BINARY)
print(binary, np.max(binary), np.min(binary))


_,contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method= cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(origin_image, contours, -1, (0,0,255), 1)

plt.imshow(img)
plt.show()