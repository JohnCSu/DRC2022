'''
README:

This Function takes in the img input and network and outputs the bounding box of the object

Input: Image from cv2.vid.read()
Optional: Network object (tensorflow, pytorch etc etc)

Output:
list of bounding boxes as tuples(x,y,w,h) to avoid.

Bounding Box coordinates as a tuple (x,y,w,h)
where x,y are coordinates for the upper left corner of the bounding box
w,h are the width and height of the box respectively

If 2 objects detected:

return [(x1,y1,w1,h1),(x2,y2,w2,h2)]

If 1 object detected still return a list of tuples:

e.g. [(x3,y3,w3,h3)]

Feel free to add more inputs and packages.
'''
import cv2
import numpy as np


def detect_object(img,net):
    #Add code here:
    return img



#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    vid = cv2.VideoCapture(0) # Capture From camera
    ret,frame = vid.read() #Get img fram from camera
    cv2.imshow(frame)
    
    print('Hello World')

    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    # while(True):
    # # Capture the video frame
    # # by frame
    #     ret,frame = vid.read()
    #     cv2.imshow(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break