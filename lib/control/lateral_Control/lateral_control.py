import cv2
import numpy as np
from numpy.linalg import norm
from math import acos,atan,sin

def purePursuit(height,width,L,l_cam,target_point):
    '''
    l: wheel-to-wheel length
    l_cam: max distance of rear_wheel to end of ROI
    
    assume that target point is always at edge of ROI
    assume that halfway of image is
    Target point: tuple (x,y)
    '''

    #Get mid point of image
    mid_point = (width/2,height)

    #Calculate ld length (right angle triangle)
    ld = norm( [l_cam,(mid_point[0]-target_point[0])] )
    alpha = acos(l_cam/ld)

    return int(np.sign(mid_point[0] - target_point[0])* atan(2*L*sin(alpha)/ld)*180/np.pi)