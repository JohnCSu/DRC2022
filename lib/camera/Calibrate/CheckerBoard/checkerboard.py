import cv2
from matplotlib.pyplot import grid
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import os
import pickle
#This will get a video feed of the camera and calibrate


def getCheckboardImages(path = './'):
    vid =cv2.VideoCapture(1)
    print('starting in:\n')

    start = time.perf_counter()
    i = 0
    while(i < 6):
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
    while(imgName < 20):
        ret,frame = vid.read()
        cv2.imshow('screenshot',frame)
        if time.perf_counter()-start > 2.0:
            cv2.imwrite(path + str(imgName)+'.png',frame)
            print(f'{imgName} has been taken')
            imgName += 1
            start = time.perf_counter()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    print('DONE!')

 

def calibrateFromImages(path = './',viewImgs = True): 
# The source for this code can be found below:
# 
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    
    grid_size = (7,5)
# termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(path+'*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            if viewImgs:
                cv2.drawChessboardCorners(img, grid_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    if viewImgs:
        cv2.destroyAllWindows()

    #Return Distortion Matrix
    ret, mtx , dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1], None,None)
    return mtx,dist
    
def CompareImages(path,mtx,dist,OptMtxCam = True):
    #Compares images between distorted and undistorted
    images = glob.glob(path+'*.png')
    img = cv2.imread(images[0])
    h,w = img.shape[:2]
    if OptMtxCam: #Calibration black magic
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w,h))
        x,y,w,h = roi
        dst = cv2.undistort(img,mtx,dist,None,newCameraMtx)[y:y+h,x:x+w]
    else:
        dst = cv2.undistort(img,mtx,dist,None)
    cv2.imshow('Undistorted',dst)
    cv2.imshow('Orig',img )
    cv2.waitKey(0)


if __name__ == '__main__':
    print(__file__)
    path = './'
    
    # print(path)
    # getCheckboardImages(path)
    
    # mtx,dist = calibrateFromImages(path,False)
    
    # with open('calibration_params','wb') as f:
    #     pickle.dump((mtx,dist),f)
    # print(__file__)
    
    with open('calibration_params','rb') as f:
        mtx,dist = pickle.load(f) 
    vid = cv2.VideoCapture(1)
    # CompareImages(path,mtx,dist)
    while(1):
        ret, img = vid.read()
        img2 = cv2.undistort(img,mtx,dist)
        cv2.imshow('orig',img)
        cv2.imshow('undistort',img2)
        cv2.waitKey(1)
