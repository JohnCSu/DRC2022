import cv2
from .lateral_Control.lateral_control import purePursuit
from .fitCurve.fitCurve import *
from matplotlib import pyplot as plt
import numpy as np
import os


class control():
    def __init__(self,w2w_dist = 26.5,w2e_dist = 147,cam_w = 640,cam_h = 480):
        self.cam_width = cam_w
        self.cam_height = cam_h
        self.wheel_to_wheel_dist = w2w_dist
        self.wheel_to_edge_dist =w2e_dist

    def Find_lane(self,*imgs):
        return findLane(*imgs)

    def Find_target_point(self,w,lanes):
        return findTargetPoint(w,lanes[0:2])


    def PurePursuit(self,h,w,target_point):
        return purePursuit(h,w,self.wheel_to_wheel_dist,self.wheel_to_edge_dist, target_point)

    
    def Decision(self,data):
        '''
        data is a dic that contains information
        '''
        h,w = data['blue_lane'].shape

        lanes = self.Find_lane(data['blue_lane'],data['yellow_lane'])
        target_point = self.Find_target_point(w,lanes)
        # print(target_point)
        angle = self.PurePursuit(h,w,target_point)
        speed = 155 #CHANGE to function of angle
        return angle,speed,target_point


if __name__=='__main__':
    img = cv2.imread('test.jpg')
    plt.imshow(img)
    plt.show()

