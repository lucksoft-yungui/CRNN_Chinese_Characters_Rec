import cv2
import numpy as np

mean = 0.588
std = 0.193
inp_w = 160
inp_h = 32

img = cv2.imread("./image-000000006.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_h, img_w = img.shape

img = cv2.resize(img, (0,0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv2.INTER_CUBIC)


cv2.imshow('img', img)
cv2.waitKey(0)

img = np.reshape(img, (inp_h, inp_w, 1))

img = img.astype(np.float32)
img = (img/255. - mean) / std
img = img.transpose([2, 0, 1])


# 调整到0-255的范围并转换为uint8
img_disp = ((img + 1) / 2. * 255.).astype(np.uint8)
img_disp = img_disp.transpose([1, 2, 0])
img_disp = np.squeeze(img_disp)

cv2.imshow('img', img_disp)
cv2.waitKey(0)

