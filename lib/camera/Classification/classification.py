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


def turnDirection(img,orb):
    #Convert to gray scale

    


    return 0

#Run your testing in the following if statement:
#The code in this if statement will only run when you run the script (it wont run when you import this)
if __name__ == '__main__':
    #Feel free to modify where fit
    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW) # Capture From camera
    ret,frame = vid.read() #Get img fram from camera

    #model = keras.models.load_model('keras_model.h5')
    
    size = (224,224)

    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    while(True):
    # Capture the video frame
    # by frame
        ret,frame = vid.read()
        cv2.imshow('og',frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        gray_filter = cv2.inRange(gray,0,100)
        cv2.imshow('og',gray_filter)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break