#Function should execute control and send a state to 
from camera.camera import camera
from control.control import control
from microcontroller.send_serial import arduino
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sys import platform
import argparse

#Argument Parser
CALIBRATE = False
TEST = False
SHOW = False
RES_DEFAULT = '480p'

grad_p_dict ={
    '480p': np.float32([[181,185],[436,188],[103,225],[506,231]]),
    '720p': None,
    '960p': None
}
roi_dict ={
    '480p': np.float32([[0,210],[639,210],[0,450],[639,450]]),
    '720p': None,
    '960p': None
}



roi = roi_dict[RES_DEFAULT]
grad_p = grad_p_dict[RES_DEFAULT]


parser = argparse.ArgumentParser()
parser.add_argument('-c','--calibrate',help = 'Calibration Flag',type = bool, choices= [True ,False])
parser.add_argument('-t','--test', help = 'Test Flag default False',type = bool, choices= [True, False])
parser.add_argument('-s','--show', help = 'Show Video Stream Flag default False',type = bool, choices= [True, False])
parser.add_argument('-r','--resolution',help ='Enter 480p,720p,960p',type = str, choices= ['480p','720p','960p'])
args = parser.parse_args()

if args.calibrate:
    CALIBRATE = True
if args.test:
    TEST = True
if args.show:
    SHOW = True
if args.resolution:
    roi = roi_dict[args.resolution]
    grad_p = grad_p_dict[args.resolution]





print(f'Calibratoin is {"on" if CALIBRATE else "off"}\n \
Testing is {"on" if TEST else "off"} \n')
print(f'Showing Img feed is {"on" if TEST else "off"} \n' ) 





if platform == 'linux':
    usb_name = '/dev/ttyUSB0'
    cam_num=0
else:
    usb_name = 'COM5'
    cam_num=1

cam_num =1

if __name__ == '__main__':
    if CALIBRATE:
        img = plt.imread('480P_cal.jpg')
        mask = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask = (255*((img + 1) > 0)).astype(np.uint8)
        cam = camera(cam_num)
        cam.setROI(grad_p=grad_p,roi_p= roi)
        mask = cam.birdsEye(mask)
        plt.imsave('corner.png',mask)
        mask = cv2.Canny(mask,50,255)
        # mask[:,0] = 255
        # mask[:,-1] = 255
        cv2.imwrite('diag.png',mask)
        plt.imshow(mask,'gray')
        plt.show()
        
        exit()

    if TEST:
        data = input('Enter g to start')
        if data != 'g':
            exit()

    out = cv2.VideoWriter('Drive.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (640,480))
    

    cam = camera(cam_num)
    cam.setROI(grad_p=grad_p,roi_p= roi)
    ret,img = cam.read()
    img2 = cam.birdsEye(img)
    
    
    w,h = img2.shape[1],img2.shape[0]
    bird_out = cv2.VideoWriter('Birdseye.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (w,h))
    lane_out = cv2.VideoWriter('Lanes.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10, (w,h))
    ctrl = control()
    # ard = arduino(port = usb_name) #Auto start the
    f = []
    while(1):
        try:
            ret,img = cam.read()
            f.append(img)
            data = cam.GetCameraData(img)
            angle,speed,state,target_points,grid = ctrl.Decision(data)
            # ard.sendData(angle,speed = 100,state = state)

            if TEST:
                img2 = cam.birdsEye(img)
                if SHOW:    
                    cv2.imshow('main',img)
                    cv2.imshow('blue',grid)
                    for target_point in target_points:
                        cv2.circle(img2,target_point,radius =10,color = (0,255,0),thickness =5 )
                    cv2.imshow('target point', img2)
                    # print(angle)
                    k = cv2.waitKey(50)
                    if k%256 == 27:
                    # ESC pressed
                        print("Escape hit, closing...")
                        break
                bird_out.write(img2)
                out.write(img)
                lane_out.write(cv2.bitwise_or(data['blue_lane'],data['yellow_lane']))
               
                # print(data.keys())
        except KeyboardInterrupt:
            print('\nTERMINATING\n')
            # ard.sendData(state = 5)
            exit()

    if TEST:
        cv2.destroyAllWindows()
        out.release()
        bird_out.release()
        lane_out.release()
    # ard.sendData(state = 0)