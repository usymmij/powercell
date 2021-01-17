import cv2
import numpy as np

def checkIfYellow(img):
    yc = 5
    xc = 5
    for column in img:
        yc += 1
        if yc < 5:
            continue
        yc = 0
        for pixel in column:
            xc += 1  
            if xc < 5:
                continue
            xc = 0
            if pixel[0] < 200 and pixel[1] > 50 and pixel[2] > 50:
                if float( pixel[1]) / float(pixel[2]) > 0.65 and float( pixel[1]) / float(pixel[2]) < 1.45:
                    if float( pixel[0]) / float(pixel[2]) < 0.7:
                        return True
    return False

cap = cv2.VideoCapture('data/a.mp4')

succ, im = cap.read()

#720, 1280
#80

imgs = []
imgc = []

while succ:
    for i in range(8): # 720/80 = 9
        for b in range(2):
            for j in range(15): # 1280/80 = 16
                for d in range(4):
                    crp = im[ 80*i + b*20 : 80*(i+1) + b*20, 80*j + d*20:80*(j+1) + d*20]
                    imgc.append(crp)
    succ, im = cap.read()
cap.release()
imgc = np.asarray(imgc)

print('saving')

for img in imgc:
    imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
imgs = np.asarray(imgs)
print(imgs.shape)
np.save('data/data.npy', imgs, True)
imgs = []
label = []

q = 0
for i in range(len(imgc)):
    if checkIfYellow(imgc[i]) == False:
        label.append([1,0,0])
        continue
    print('labeled: ' + str(i))
    print('labeled with ball: ' + str(q))
    cv2.imshow('a',imgc[i])
    key  = cv2.waitKey(0)
    if key == 45:
        label.append([1,0,0])
    if key == 43:
        q += 1
        label.append([0,1,0])
    if key == 13:
        q += 1
        label.append([0,0,1])
    if key == ord('~'):
        label = np.asarray(label)
        np.save('data/label.npy',label, True)
        exit()
label = np.asarray(label)
np.save('data/label.npy',label, True)
exit()