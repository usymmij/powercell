import cv2
import numpy as np

cap = cv2.VideoCapture('data/a.mp4')

succ, im = cap.read()

#720, 1280
#80

imgs = []

while succ:
    for i in range(9): # 720/80 = 9
        for j in range(16): # 1280/80 = 16
            crp = im[80*i:80*(i+1), 80*j:80*(j+1)]
            imgs.append(crp)
    succ, im = cap.read()
imgs = np.asarray(imgs)

for i in range(len(imgs)):
    cv2.imshow('a',imgs[i])
    cv2.waitKey(0)
    if i == 100:
        exit()
    