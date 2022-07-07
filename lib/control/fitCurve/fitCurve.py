import cv2
import numpy as np
from matplotlib import pyplot as plt


def findLane(*imgs,rho=2,theta = 2*np.pi/180,minLineLength=100,maxLineGap=50,threshold=100):
    '''
    Takes in a birds eye black and white canny image (1 channel) and fits a quadratic curve
    
    to each lane
    '''
    # print(imgs)
    lanes = []
    for img in imgs:
        lines = cv2.HoughLinesP(img, rho=rho, theta=theta, threshold=threshold, minLineLength = minLineLength , maxLineGap=maxLineGap)
        # lines = cv2.HoughLines(img,rho =rho,theta = theta,threshold=threshold)
        # [ np.cosl[0][0], for l in lines]
        if lines is None:
            lanes.append(np.array([np.inf,0]))
        else:
            co_ords = np.array([l[0] for l in lines])
            
            co_ords = np.reshape(co_ords, (co_ords.shape[0]*2,2) )
            line_eq = np.polyfit(co_ords[:,0],co_ords[:,1],deg = 1)
            lanes.append(line_eq)

    return lanes
    

def findTargetPoint(w,lanes):

    #Always left lane first then right lane
    p = []
    for i,lane in enumerate(lanes):
        if lane[0] == np.inf:
            if i == 0: #Left Lane
                x = 0
            else: #Right Lane
                x = w
        else:
            x = int( -(lane[1])/lane[0])
        p.append(x)

    target = ((p[0]+p[1])//2,0)
    return  target if target[0] > 0 and target[0] < w else (w//2,0)
    








if __name__ == '__main__':
    
    # vid = WebcamVideoStream(src=701)
    # vid.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # vid.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # # vid.stream.set(cv2.CAP_DSHOW)
    # # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensionns to cam object (not cap)
    #     # self.cam.set(

    # vs = vid.start()
    # fps = FPS().start()
    # frame_1 = vs.read()
    # # loop over some frames...this time using the threaded stream
    # while 1:
    #     # grab the frame from the threaded video stream and resize it
    #     # to have a maximum width of 400 pixels
        
    #     # check to see if the frame should be displayed to our screen
    #     frame = vs.read()
    #     # print(frame)
    #     cv2.imshow("Frame", frame_1)

    #     k = cv2.waitKey(1)
    #     if k%256 == 27:
    #         # ESC pressed
    #         print("Escape hit, closing...")
    #         break
            
    #     # update the FPS counter
    #     fps.update()
    # # stop the timer and display FPS information
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # # do a bit of cleanup
    # cv2.destroyAllWindows()
    # vs.stop()

    img1 = cv2.imread('test.jpg')

    img = img1[:,int(img1.shape[1]/1.5):]
    # img = img1  
    
    # print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img,100,150)
    cv2.imshow('name',img)
    cv2.waitKey(0)
    # plt.imshow(img,'gray')
    # plt.show()
    line = findLane(img)
    

    img = img1[:,int(img1.shape[1]/1.5):]
    

    #Get lines for y = 0, y = height

    def getPoints(img,line):
        #Get (x,y) tuples at top and bot of image 
        return [ (int( (y-line[1])/line[0] ),int(y)) for y in (0,img.shape[0])]
            
    p1,p2 =getPoints(img,line[0])
    print(p1,p2)
    cv2.line(img,p1,p2,color = (0,255,0),thickness= 10)


    cv2.imshow('name',img)
    cv2.waitKey(0)

    # plt.imshow()
    # plt.show()
