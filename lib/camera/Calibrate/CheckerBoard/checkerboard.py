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
    cam =cv2.VideoCapture(1,cv2.CAP_DSHOW)
    count = 0
    while(count < 20):
        ret, img = cam.read()
        
        cv2.imshow('video',img)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = f"{count}.png"
            cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            count+=1
        
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
    
    path = os.path.dirname(__file__)
    to_cal = False


    # a = plt.imread('0.png')
    # print(a.shape)
    # exit()
    # print(path)
    if to_cal:
        getCheckboardImages(path)
        
        # mtx,dist = calibrateFromImages(path,False)
        mtx,dist = calibrateFromImages()
        with open('calibration_params','wb') as f:
            pickle.dump((mtx,dist),f)
        # print(__file__)

    with open(os.path.join(path,'calibration_params'),'rb') as f:
        mtx,dist = pickle.load(f) 
    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  # set new dimensionns to cam object (not cap)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    # CompareImages(path,mtx,dist)
    
    t = 0
    f = 0
    
    SPF =int(1/30*1000)
    while(1):
        start = time.perf_counter()
        ret, img = vid.read()
        if ret:
        # img2 = cv2.undistort(img,mtx,dist)
            cv2.imshow('orig',img)
            # print(img.shape)
            # cv2.imshow('undistort',img2)
            # cv2.waitKey(1)
            if cv2.waitKey(1)%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            # time.sleep(SPF)
            t += time.perf_counter() - start
            f +=1
    
    print(f'Total Frames: {f} \t Total Time: {t}\n FPS = {f/t}')