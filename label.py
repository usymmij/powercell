import cv2
import numpy as np

cap = cv2.VideoCapture('data/a.mp4')

succ, im = cap.read()

#720, 1280
#80

imgs = []

while succ:
    for i in range(8): # 720/80 = 9
        for j in range(15): # 1280/80 = 16
            for b in range(2):
                for d in range(4):
                    crp = im[ 80*i + b*20 : 80*(i+1) + b*20, 80*j + d*20:80*(j+1) + d*20]
                    imgs.append(crp)
    succ, im = cap.read()
imgs = np.asarray(imgs)
print('saving')
print(imgs.shape)
np.save('data/data.npy', imgs, True)

label = []

for i in range(len(imgs)):
    cv2.imshow('a',imgs[i])
    key  = cv2.waitKey(0)
    if key == 43:
        label.append(0)
    if key == 13:
        label.append(1)