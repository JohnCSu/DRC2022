'''
README:

This Function takes in the img input and network and determines whether we need to take
the left lane or the right lane 

This function should output 3 possible states Left,Right or a 'do nothing' state.
The output can be anything e.g 
0 for take left lane, 
1 for take right lane,
None for 'do nothing'/no sign detected

Just be consistent and stick with how you will output the state  

Feel free to add more inputs and packages.
'''
import cv2
import numpy as np
import os 
import random

def turnDirection(img ):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Convert to gray scale and threshold
    mask = cv2.inRange(gray_img,0,100)
    kernal = np.ones((5, 5), "uint8")
    # cv2.imshow('mask',mask)
    m= cv2.dilate(mask,kernal)
    contours, _ = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        if cv2.contourArea(contour) > 5000 and cv2.contourArea(contour) < 10000:
            return 'left' if random.random() < 0.5 else 'right'
    
    return 'straight'
    



#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    vid = cv2.VideoCapture('4.mp4') # Capture From camera

    # x = [ cv2.imread(f,cv2.IMREAD_GRAYSCALE ) for f in glob('*.png')]

    # target = [cv2.resize(f,dsize=(200,200)) for f in x]
    ret,frame = vid.read() #Get img fram from camera

    #model = keras.models.load_model('keras_model.h5')
    
    size = (224,224)

    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON
    
    while(True):
    # Capture the video frame
    # by frame
        ret,frame = vid.read()
        cv2.resize(frame,(640,480),frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        turns = turnDirection(gray)
        for x,y,w,h in turns:
            cv2.rectangle(frame,(x,y),(x+w,y+h), color = (0,0,0), thickness = 4)

        cv2.imshow('og',frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break