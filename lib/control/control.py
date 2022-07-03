
from birdsEye.birdEye import birdsEye

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


class control():
    def __init__(self,src,dst):
        self.src,self.dst = src,dst
    
    def birdsEye(self,img):
        return birdsEye(img,self.src)



if __name__=='__main__':
    img = cv2.imread('test.jpg')
    plt.imshow(img)
    plt.show()

