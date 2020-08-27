import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/home/ogai/models/research/deeplab/exp/soiling_train/vis/vis/soiling1_10.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
total = 0

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [c], [255,255,255])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    pixels = cv2.countNonZero(mask)
    total += pixels
    cv2.putText(image, '{}'.format(pixels), (x ,y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

print('mask pixel',total)
total_pixel = 2073600
coverage = round((total / total_pixel),2)*100

cv2.putText(image, 'coverage of soiling part = '+'{}'.format(coverage) +'%' , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

print('coverage of soiling part ',coverage)

cv2.namedWindow("thresh",0)
cv2.resizeWindow("thresh", 1440, 900)
cv2.imshow("thresh",thresh)


#cv2.imshow('thresh', thresh)
#cv2.imshow('image', image)
#cv2.waitKey(0)

cv2.namedWindow("image",0)
cv2.resizeWindow("image", 1440, 900)
cv2.imshow("image",image)
cv2.waitKey(0)

