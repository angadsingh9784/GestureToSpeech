import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap=cv2.VideoCapture(0)
#gives the skeleton hand for better for image classifier
detector=HandDetector(maxHands=1)

offset=20
imgSize=300

folder="Data/1"
counter=0

# OPEN THE CAMERA
while True:
    success, img=cap.read()
    hands,img=detector.findHands(img)

    #now we crop the image
    if hands:
        hand=hands[0]
        #because we have only one hand
        # now we want bounding box
        x,y,w,h=hand['bbox']
        #now we can crop the image
        #dimensions
        #strting hig-ending wid
        imgWhite= np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop=img[y-offset:y + h+offset,  x-offset:x  +  w+offset]

        imgCropShape=imgCrop.shape
        #put image crop matrix in white image

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)     #tocenter the image
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # tocenter the image
            imgWhite[hGap:hCal + hGap,: ] = imgResize


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

    cv2.imshow("Image",img)
    key=cv2.waitKey(1)

    if key==ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)