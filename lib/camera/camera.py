

from camera.birdsEye.birdEye import birdsEye
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
    def __init__(self,cam_num =1,resolution = (640,480)):
        #set Camera feed and set camera dimensions:
        self.setCamera(cam_num,resolution)
        #For Lane Detection
        self.hsv_masks =  {
            'blue' : ( np.array([180//2, 63, 63]) , np.array([270//2, 255, 255])),
            'yellow':(np.array([40//2, 63, 63]),np.array([60//2, 255, 255]))
        }

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


    def setROI(self,src=None):
        #Parameters for birds eye
        if src is None:
            self.src = np.float32([[166,194],[500,194],[0,340],[639,340]])
        else:
            self.src =src
        self.persp_width = int(norm( np.subtract(self.src[2],self.src[3]))) #Gets lower right and left length (iso trap so always lower points length will be max length)
        self.persp_height =int(norm(np.subtract(self.src[0],self.src[2]))) # Gets a length (iso trapezium so any height)
        self.dst =np.float32( [
            [0,0],
            [self.persp_width-1,0],
            [0,self.persp_height-1],
            [self.persp_width-1,self.persp_height-1 ]
        ] )
        self.persp_M = cv2.getPerspectiveTransform(self.src,self.dst)
    #Returns the ret,frame of video capture
    def read(self):
        return self.cam.read()
    
    #Return calibrate function
    def calibrate(self,img):
        return calibrate(img,self.mtx,self.dist)
    
    #Return Bird's eye view img
    def birdsEye(self,img):
        return birdsEye(img,self.persp_width,self.persp_height,self.persp_M)

    def detect_lane(self,img):
        return detect_lane(img,self.hsv_masks)
    #Return all the camera data needed for control
    def GetCameraData(self,img):
        img = self.calibrate(img)
        birdeye = self.birdsEye(img)
        b_lane,y_lane = detect_lane(birdeye,self.hsv_masks)
        turn= None
        obstacle = None
        return {'turn':turn,
                'obstacle': obstacle,
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
            
    
    