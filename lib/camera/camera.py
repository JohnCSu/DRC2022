

from camera.birdsEye.birdEye import birdsEye,getROIwarp
from camera.Calibrate.calibrate import calibrate 
from camera.Classification.classification import turnDirection
from camera.ObjectDetection.object_detection import detect_object
from camera.LaneDetection.lane_detection import detect_lane
from camera.Calibrate.CheckerBoard import checkerboard as checkers
import pickle
import numpy as np
from numpy.linalg import norm
import cv2
import os
from matplotlib import pyplot as plt 
from sys import platform
#Executes all the camera stuff and returns a dictionary for the state to control

class camera():
    def __init__(self,cam_num =1,resolution = (640,480),gamma = 3):
        #set Camera feed and set camera dimensions:
        self.setCamera(cam_num,resolution)
        #For Lane and Object Detection
        self.hsv_masks =  {
            'blue' : ( np.array([180//2, 63, 63]) , np.array([270//2, 255, 255])),
            'yellow':(np.array([40//2, 10, 63]),np.array([70//2, 255, 255]))

        }
        self.obj_masks = {
            'object': (np.array([135, 87, 111], np.uint8),np.array([180, 255, 255], np.uint8)),
            'object2': (np.array([0, 87, 111], np.uint8),np.array([10, 255, 255], np.uint8)),
            'green': (np.array([35, 52, 72], np.uint8),np.array([82, 255, 255], np.uint8)),
                }


        self.gamma = gamma
        self.lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            self.lookUpTable[0,i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)


        #For Calibration
        cal_path = os.path.join(os.path.dirname(checkers.__file__ ),'calibration_params')
        with open(cal_path,'rb') as f:
            self.mtx,self.dist = pickle.load(f)

        #For Birds Eye view
        self.setROI()
    
    def setCamera(self,cam_num,resolution):
        self.cam_num = cam_num
        if isinstance(cam_num,str):
            self.cam = cv2.VideoCapture(cam_num)
        else:
            if platform == 'win32':
                self.cam = cv2.VideoCapture(cam_num,cv2.CAP_DSHOW)
            else:
                self.cam = cv2.VideoCapture(cam_num)
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # set new dimensionns to cam object (not cap)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.width  = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`


    def setROI(self,grad_p = np.float32([[200,153],[445,153],[0,351],[639,351]]),roi_p = np.float32([[0,135],[639,135],[0,450],[639,450]])):
        self.grad_p = grad_p
        self.roi_p = roi_p
        #Parameters for birds eye
        dst,(w,h) = getROIwarp(grad_p,roi_p) 
        self.persp_width = w
        self.persp_height = h
        
    # corrected = cv2.convertScaleAbs(img,alpha = 0.8,beta = -50)
    # persp_M = cv2.getPerspectiveTransform(grad_p,dst)
        self.persp_M = cv2.getPerspectiveTransform(roi_p,dst)
        
    #Returns the ret,frame of video capture
    def read(self):
        return self.cam.read()
    
    #Return calibrate function
    def calibrate(self,img):
        return calibrate(img,self.mtx,self.dist)
    
    #Return Bird's eye view img
    def birdsEye(self,img):
        return birdsEye(img,self.persp_width,self.persp_height,self.persp_M)

    def detect_object(self,img):
        return detect_object(img,self.obj_masks,roi_h = (int(self.roi_p[0][1]),int(self.roi_p[2][1])))
    def detect_lane(self,img):
        return detect_lane(img,self.hsv_masks,self.lookUpTable)
    #Return all the camera data needed for control
    def GetCameraData(self,img):
        img = self.calibrate(img)
        
        birdeye = self.birdsEye(img)
        b_lane,y_lane = self.detect_lane(birdeye)
        obstacles = self.detect_object(birdeye)
        turn= None
        return {'turn':turn,
                'obstacle': obstacles['object'],
                'green' : obstacles['green'],
                'blue_lane': b_lane,
                'yellow_lane':y_lane
                }





if __name__ == '__main__':
    # cam_num = 'test2.mp4'
    cam_num = 'test480p.mp4'
    cam = camera(cam_num)
    if isinstance(cam_num,str):
        cond = cam.cam.isOpened
    else:
        cond = 1
    
    # print(cam.mtx)
    setpoints = False
    # img = cv2.imread('test.jpg')
    # img = cam.calibrate(img)


    # im = plt.imread('test.jpg')
    # im = cam.calibrate(im)
    # plt.imshow(cam.birdsEye(im))
    # plt.show()

    # exit()

    if setpoints:
        while(cond):
            ret,img = cam.read()
            print(type(img))
            plt.imshow(img)
            plt.show()
            exit()
    
    print(cam.mtx,cam.dist)
    print(cam.height,cam.width)
    while(cond):
        ret, img = cam.read()
        img2 = cam.calibrate(img)
        cv2.imshow('orig',img)
        cv2.imshow('undistort',img2)
        bird = cam.birdsEye(img2)
        # cv2.imshow('birdeys',bird)
        edge = cam.detect_lane(bird)
        cv2.imshow('lane Birds eye View',cv2.bitwise_or(edge[0],edge[1]))
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
            
    
    