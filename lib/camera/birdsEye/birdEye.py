from webbrowser import WindowsDefault
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import os
def birdsEye(img,width,height,M):

    return cv2.warpPerspective(img,M,(width, height),flags=cv2.INTER_LINEAR) 


def getROIwarp(grad_p,roi_p ):
    #Region is a tuple (y1,y2)

    w,h = int(roi_p[1][0] - roi_p[0][0]),  int(roi_p[2][1] - roi_p[0][1])

    #Calc average gradient (assume iso)
    dydxL = (grad_p[0][1] - grad_p[2][1])/(grad_p[0][0] - grad_p[2][0])
    dydxR = (grad_p[1][1] - grad_p[3][1])/(grad_p[1][0] - grad_p[3][0])
    
    dydx = (abs(dydxL) + abs(dydxR))/2
    

    p =(roi_p[0][1]-grad_p[0][1])/(-dydx) + grad_p[0][0]
    p2 = (roi_p[2][1]-grad_p[2][1])/(-dydx) + grad_p[2][0]
    
    dx =abs(p)
    width =int(w+2*dx)
    # width = 1200
    t = p + 20 #-abs(p2) #Tuning Factor idk
    warp_points = [[0,0],[2*dx+w,0],[dx+t,h],[dx+w-t,h]] 
    return np.float32(warp_points),(width,h)


if __name__ == '__main__':
    img = cv2.imread('480P_cal.jpg')
    #Gradients
    grad_p = np.float32([[217,150],[431,150],[0,374],[639,374]])
    
    roi_p = np.float32([[0,110],[639,110],[0,450],[639,450]])

    dst,(w,h) = getROIwarp(grad_p,roi_p) 
    # persp_M = cv2.getPerspectiveTransform(grad_p,dst)
    persp_M = cv2.getPerspectiveTransform(roi_p,dst)
    
    print(np.matmul(persp_M,np.array([217,150,1])))
    # M_off = np.array([ [ 1 , 0 , -200],[ 0 , 1 , 0],[ 0 , 0 ,1]])
    # persp_M = np.matmul(M_off,persp_M)
    img2 = birdsEye(img,w,h,persp_M)
    # img2 = birdsEye(img,640,440-135,persp_M)
    # cv2.imshow('Og',img)
    plt.imshow(img2)
    plt.show()
    # cv2.waitKey(0)