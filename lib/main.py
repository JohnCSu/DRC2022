#Function should execute control and send a state to 
from camera.camera import camera
from control.control import control
from microcontroller.send_serial import arduino
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import platform
import argparse
import keyboard

#Argument Parser
CALIBRATE = False
TEST = False
RES = '480p'


parser = argparse.ArgumentParser()
parser.add_argument('-c','--calibrate',help = 'Calibration Flag',type = bool, choices= [True ,False])
parser.add_argument('-t','--test', help = 'Test Flag default False',type = bool, choices= [True, False])
args = parser.parse_args()
if args.calibrate:
    CALIBRATE = True
if args.test:
    TEST = True

print(f'Calibratoin is {"on" if CALIBRATE else "off"}\n \
Testing is {"on" if TEST else "off"} \n') 
cal_480p = np.float32([[200,153],[445,153],[0,351],[639,351]])




if platform == 'linux':
    usb_name = '/dev/ttyUSB0'
else:
    usb_name = 'COM5'
    



if __name__ == '__main__':
    if CALIBRATE:
        img = plt.imread('480P_cal.jpg')
        plt.imshow(img)
        plt.show()
        exit()

    if TEST:
        data = input('Enter g to start')
        if data != 'g':
            exit()

    out = cv2.VideoWriter('Drive.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (640,480))
    

    cam = camera(cam_num=0)
    ret,img = cam.read()
    img2 = cam.birdsEye(img)
    w,h = img2.shape[1],img2.shape[0]
    bird_out = cv2.VideoWriter('Birdseye.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (w,h))
    lane_out = cv2.VideoWriter('Lanes.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (w,h))
    cam.src = cal_480p
    ctrl = control()
    ard = arduino(port = usb_name) #Auto start the
    f = []
    while(1):
        try:
            ret,img = cam.read()
            f.append(img)
            data = cam.GetCameraData(img)
            angle,speed,target_point = ctrl.Decision(data)
            ard.sendData(angle,speed = 100,state = 1)

            if TEST:
                img2 = cam.birdsEye(img)
                
                cv2.imshow('main',img)
                cv2.imshow('blue',data['blue_lane'])
                cv2.imshow('yellow',data['yellow_lane'])
                cv2.circle(img2,target_point,radius =10,color = (0,255,0),thickness =5 )
                cv2.imshow('target point', img2)


                bird_out.write(img2)
                out.write(img)
                lane_out.write(cv2.bitwise_or(data['blue_lane'],data['yellow_lane']))
                print(angle,target_point)
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                # print(data.keys())
        except KeyboardInterrupt:
            ard.sendData(state = 0)
            exit()

    if TEST:
        cv2.destroyAllWindows()
        out.release()
    ard.sendData(state = 0)