'''
README:

This Function takes in the img input and return the left 'blue' lane filter image and 
rright 'yellow' lane filter image as 2xhxw numpy array

Might just add Green 'finish/start' filter as well

Feel free to add more inputs and packages.
'''
import cv2
import numpy as np


def detect_lane(img,hsv_masks):
    blur_frame = cv2.GaussianBlur(frame,(5,5),0)
    #Convert to HSV color scheme
    HSV_f = cv2.cvtColor(blur_frame,cv2.COLOR_BGR2HSV)
    #Hue Range is 0 to 180
    #Detect Blue for now


    blue = cv2.inRange(HSV_f,hsv_masks['blue'][0],hsv_masks['blue'][1])
    yellow = cv2.inRange(HSV_f,hsv_masks['yellow'][0],hsv_masks['yellow'][1])

    return np.array([cv2.Canny(mask,50,150) for mask in [blue,yellow]])
    

#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    vid = cv2.VideoCapture(0) # Capture From camera
    ret,frame = vid.read() #Get img fram from camera

    masks = {
    'blue' : ( np.array([180//2, 63, 63]) , np.array([270//2, 255, 255])),
    'yellow':(np.array([40//2, 63, 63]),np.array([60//2, 255, 255]))
    }
    img = detect_lane(frame,masks)
    
    # cv2.imshow(img)
    

    print('Hello World')

    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    # while(True):
    # # Capture the video frame
    # # by frame
    #     ret,frame = vid.read()
    #     cv2.imshow(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break