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


def detect_object(img,hsv_masks,area = [400,4000]):

    hsv_f = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = {}
    obj_d = {'object' : [],'green' : []}
    mask['object'] = [cv2.bitwise_or(cv2.inRange(hsv_f,hsv_masks['object'][0], hsv_masks['object'][1]),cv2.inRange(hsv_f,hsv_masks['object2'][0], hsv_masks['object2'][1])),area[0]]
    mask['green'] = [cv2.inRange(hsv_f,hsv_masks['green'][0], hsv_masks['green'][1]),area[1]]

    # mask['object'][0] = mask['object'][0][210:450,:]  
    # mask['green'][0] = mask['green'][0][210:450,:]
    kernal = np.ones((9, 9), "uint8")

    for col,(mask,area_threshold) in mask.items():
        m = cv2.dilate(mask,kernal)
        contours, hierarchy = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > area_threshold):
                x, y, w, h = cv2.boundingRect(contour)
                obj_d[col].append((x,y,w,h))

    return obj_d



#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW) # Capture From camera
    # ret,frame = vid.read() #Get img fram from camera
    # cv2.imshow(frame)
    
    # print('Hello World')

    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    hsv_masks = {
        'object': (np.array([130, 50, 50], np.uint8),np.array([180, 255, 255], np.uint8)),
        'object2': (np.array([0, 87, 111], np.uint8),np.array([10, 255, 255], np.uint8)),
        'green': (np.array([45, 52, 50], np.uint8),np.array([82, 255, 255], np.uint8)),
        
    }
    while(True):
    # Capture the video frame
    # by frame
        ret,frame = vid.read()
        detect = detect_object(frame,hsv_masks)

        for o in detect['object']:
            x1,y1,w,h = o
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),color = [0,0,255],thickness = 4)
        for o in detect['green']:
            x1,y1,w,h = o
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),color = [0,255,0],thickness = 4)
        cv2.imshow('Hi',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break