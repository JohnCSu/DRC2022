import cv2
from .lateral_Control.lateral_control import purePursuit
from .fitCurve.fitCurve import *
from matplotlib import pyplot as plt
import numpy as np
import os
import time

class control():
    def __init__(self,w2w_dist = 26.5,w2e_dist = 198,cam_w = 640,cam_h = 480):
        self.cam_width = cam_w
        self.cam_height = cam_h
        self.wheel_to_wheel_dist = w2w_dist
        self.wheel_to_edge_dist =w2e_dist
        
        self.block_size = (10,10)
        
        path = os.path.join(os.path.dirname(__file__ ),'diag.png')
        self.diag  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        self.start = True
        self.state = 0
        self.foundGreen = False

    def getPath(self, blue,yellow,objects):
        return getpath(blue,yellow,objects,self.block_size,self.diag)

    def PurePursuit(self,h,w,target_point):
        return purePursuit(h,w,self.wheel_to_wheel_dist,self.wheel_to_edge_dist, target_point)

    def findstartLine(self,green):
        
        if len(green) > 0:
            if self.state == 0:
                return 1
            elif self.state == 1:
                return 2

    def Decision(self,data):
        '''
        data is a dic that contains information
        '''
        h,w = data['blue_lane'].shape
        if self.start:
            cv2.resize(self.diag,(w,h),self.diag)
            self.start = False

        if not self.foundGreen:
            state = self.findstartLine(data['green'])
            if state != self.state: #If we have a change in state
                self.start = time.perf_counter()
                self.state = state
                self.foundGreen = True
        else:
            if time.perf_counter() - self.start > 3.0:
                self.foundGreen = False

        path_points,grid = self.getPath(data['blue_lane'],data['yellow_lane'],data['obstacle'])
        # print(target_point)
        angle = self.PurePursuit(h,w,path_points)
        speed = 110 #CHANGE to function of angle
        return angle,speed,self.state,path_points,grid


if __name__=='__main__':
    img = cv2.imread('test.jpg')
    plt.imshow(img)
    plt.show()

