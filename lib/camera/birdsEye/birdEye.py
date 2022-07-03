from string import hexdigits
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import os
def birdsEye(img,width,height,M):
    '''
    Input
    img: numpy array
    src: tuple of 4 coordinates (numpy array) that define our region of interest
    dst: tuple of 4 coordinates that our src points map to

    Order:
    src(0) : top-left corner
    src(1) : top-right corner
    src(2) :bot-left corner
    src(3) : bot-right corner

    '''
    return cv2.warpPerspective(img,M,(width, height),flags=cv2.INTER_LINEAR) 

if __name__ == '__main__':
    #Set True to calibrate points
    pass
    # x = [[140, 210],[400, 210],[0, 350],[639, 410]]
    # src =tuple(x)
    # print(src)
    # print(os.getcwd())
    # img = cv2.imread('3.png')
    # birdsEye(img,src)
    # plt.imshow(img)
    # plt.show()