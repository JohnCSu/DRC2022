import cv2
import numpy as np
from numpy.linalg import norm
from math import acos,atan,hypot,sin

def purePursuit(height,width,w2w,l_cam,target_points):
    '''
    l: wheel-to-wheel length
    l_cam: max distance of rear_wheel to end of ROI
    
    assume that target point is always at edge of ROI
    assume that halfway of image is
    Target point: tuple (x,y)
    '''
    
    #Split Data up into Thirds and average and weight them
    x,y = target_points[:,0],target_points[:,1]
    split = len(x)//4
    vert = [-(146)/(450-210), l_cam]
    hor = 50/163.2
    #Order is from bottom (h) to top (0)
    angle = np.zeros(4)
    for i in range(4):
        '''
        46 cm at 450,150 at 210 edge
        '''
        xt,yt = x[i*split:(i+1)*split],y[i*split:(i+1)*split]
        if len(xt) > 0:
            xt,yt = np.mean(xt),yt[-1]
        else:
            xt,yt = width//2,height//(i+1)
        mid_point = (width/2,yt)
        l_cam,w_cam = vert[0]*yt + vert[1],hor*(mid_point[0]-xt)
    
        ld = hypot(l_cam,w_cam)
        
        
        alpha = atan(w_cam/l_cam)
        # print(l_cam,w_cam,alpha*180/np.pi)

        angle[i] = (np.sign(mid_point[0] -xt)* atan(2*w2w*sin(alpha)/(0.7*ld))*180/np.pi)
        
    return int(np.dot(angle,np.array([3,3,2,1])/6))
    
    return int(np.mean(angle))