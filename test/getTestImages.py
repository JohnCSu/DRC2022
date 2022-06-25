import cv2
import numpy as np
import os
import time

def getTestImages(folder = 'laneDetection',numImages = 20,img_delay = 1.5):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    path = os.getcwd()+f'\\{folder}\\'
    print(path)
    vid =cv2.VideoCapture(1)
    print('starting in:\n')
    
    start = time.perf_counter()
    i = 0

    #6 second Countdown to get you ready
    while(i < 3):
        ret,frame = vid.read()
        cv2.imshow('start',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.perf_counter()- start > 1:
            print(i)
            i += 1
            start = time.perf_counter()


    imgName = 0
    start = time.perf_counter()
    while(imgName < numImages):
        ret,frame = vid.read()
        cv2.imshow('screenshot',frame)
        if time.perf_counter()-start > img_delay:
            cv2.imwrite(path + str(imgName)+'.png',frame)
            print(f'{imgName} has been taken')
            imgName += 1
            start = time.perf_counter()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    print('DONE!')

if __name__ == '__main__':
    #Set this folder as root    
    print(os.getcwd())

    getTestImages(folder = 'classification',numImages = 10)