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
    vid = cv2.VideoCapture(0) # Capture From camera
    ret,frame = vid.read() #Get img fram from camera

    #model = keras.models.load_model('keras_model.h5')
    
    size = (224,224)

    data = np.ndarray(shape = (1,224,224,3))
    ##UNCOMMENT TO HAVE IMAGE FEED TO TEST YOUR FUNCTION ON

    while(True):
    # Capture the video frame
    # by frame
        ret,frame = vid.read()
        cv2.imshow('og',frame)
        resized =cv2.resize(frame,size)
        normalized_image_array = (resized.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        cv2.imshow('resize',data[0])
        prediction = model.predict(data)
        print(prediction)
        
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break