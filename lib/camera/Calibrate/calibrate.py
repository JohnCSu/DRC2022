'''
README:

This Function takes in the war img input from the camera and correctly applys a calibration filter

Feel free to add more inputs and packages.
'''
import cv2
import numpy as np


def calibrate(img,mtx,dist):
    return cv2.undistort(img,mtx,dist,None)

#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    # vid = cv2.VideoCapture(0) # Capture From camera
    # ret,frame = vid.read() #Get img fram from camera

    pass
    # print('Hello World')

    # ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    # # while(True):
    # # # Capture the video frame
    # # # by frame
    # #     ret,frame = vid.read()
    # #     cv2.imshow(frame)
    # #     if cv2.waitKey(1) & 0xFF == ord('q'):
    # #         break