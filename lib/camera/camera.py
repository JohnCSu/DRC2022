import Calibrate.CheckerBoard.checkerboard as checkers
from birdsEye.birdEye import birdsEye
from Calibrate.calibrate import calibrate 
from Classification.classification import turnDirection
from ObjectDetection.object_detection import detect_object
from LaneDetection.lane_detection import detect_lane
import pickle
import numpy as np
from numpy.linalg import norm
import cv2
import os
from matplotlib import pyplot as plt 

#Executes all the camera stuff and returns a dictionary for the state to control

class camera():
    def __init__(self,cam_num =1):
        #set Camera feed and set camera dimensions:
        self.setCamera(cam_num)
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
    
    def setCamera(self,cam_num):
        self.cam_num = cam_num
        self.cam = cv2.VideoCapture(cam_num)
        self.width  = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`


    def setROI(self,src=None):
        #Parameters for birds eye
        if src is None:
            self.src = np.float32([[201,232],[472,232],[29,347],[622,347]])
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

    #Return all the camera data needed for control
    def GetCameraData(self,img,turn_net,obj_net):
        img = calibrate(img)
        # turn = turnDirection(img,turn_net)
        # obstacle =detect_object(img,obj_net )
        b_lane,y_lane = None,None# detect_lane(img,self.hsv_masks)
        turn,obstacle = None,None
        return {'turn':turn,
                'obstacle': obstacle,
                'blue_lane': b_lane,
                'yellow_lane':y_lane
                }



if __name__ == '__main__':
    cam = camera()
    # print(cam.mtx)
    setpoints = False
    # img = cv2.imread('test.jpg')
    # img = cam.calibrate(img)
    
    
    # CompareImages(path,mtx,dist)
    while(1):
        ret, img = cam.read()
        img2 = cam.calibrate(img)
        cv2.imshow('undistort',img2)
        bird = cam.birdsEye(img2)
        cv2.imshow('birdeys',bird)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "2.png"
            cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            break
            
    if setpoints:
        img =plt.imread('1.png')
        plt.imshow(img)
        plt.show()
    