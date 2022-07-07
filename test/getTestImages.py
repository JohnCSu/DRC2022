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


def takePhoto():
    #OPens up a window allowing you to take a photo
    #By pressing SPACEBAR and saves it
    cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    while(1):
        ret, img = cam.read()
        # print(ret)
        cv2.imshow('window',img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "test.png"
            cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            break
if __name__ == '__main__':
    #Set this folder as root    
    print(os.getcwd())
    takePhoto()
    # getTestImages(folder = 'classification',numImages = 10)