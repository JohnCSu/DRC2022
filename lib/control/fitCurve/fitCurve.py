from inspect import getmembers
from os import stat
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import isinf

from skimage.util import view_as_blocks

def findStart(lane,i_start ,side = 'right',block_size = (40,40),threshold = 20):

    b_w,b_h  = block_size
    h,w = lane.shape
    pad_w = block_size[0] - (w % block_size[0])
    pad_h =block_size[1] - (h % block_size[1])

    lane = np.pad(lane,((0,pad_h),(0,pad_w)))
    block_view = view_as_blocks(lane,block_size)

    flatten_view = block_view.reshape(block_view.shape[0], block_view.shape[1], -1)

    mean_view = ( np.mean(flatten_view, axis=2))

    # idx = np.argwhere(mean_view > threshold ).T
    mean_view = 255*(mean_view > threshold).astype(np.uint8)
    idx = np.nonzero(mean_view)


    #Cases if can't find Starting point
    if  len(idx) == 0: #Check if empty:
        return cv2.resize(src = mean_view,dsize = lane.shape[::-1],interpolation = cv2.INTER_NEAREST)[0:h,0:w],(h,i_start)
    elif idx[0].size == 0:
        return cv2.resize(src = mean_view,dsize = lane.shape[::-1],interpolation = cv2.INTER_NEAREST)[0:h,0:w],((h,i_start))

    if side == 'right':
        x = np.argmax(idx[0])
        # print(x)
        #Of the last row, we want
        i,j = idx[0][x],idx[1][x]
    else:
        i,j = idx[0][-1],idx[1][-1]
    # print(i,j)

    for k in range(mean_view.shape[0] - i):
        mean_view[i+k,j] = 255
        #Slightly not correct but should be fine
    return cv2.resize(src = mean_view,dsize = lane.shape[::-1],interpolation = cv2.INTER_NEAREST)[0:h,0:w],(i*b_h,j*b_w)


def SetMasks(lanes,diag,starting_points):
    
    y1,x1,y2,x2 = starting_points    
    grid = cv2.bitwise_or(lanes[0][0],lanes[1][0])
    cv2.bitwise_or(grid,diag,grid)
    grid[min(y1,y2):,x1+1:x2-1] = 0
    # print(y1,x1,y2,x2)
    return grid
 


def getMidPoint(grid,starting_points,num_points = 20, endpoint = None):
    #Should be a merged 
    #Start from bot and work your way up
    h,w = grid.shape
    
    #Basically FizzBang

    x1 = starting_points[0][1]
    x2 = starting_points[1][1]
    mid_point = (x1+x2)//2

    step = h//num_points

    if endpoint is None:
        endpoint = h
        
    for i,row in enumerate(grid[h-1::-step]): #Start from h to 0
        boundary = np.nonzero(row)[0] 
        
        boundary -= mid_point

        if boundary[boundary < 0].shape[0]:
            #Find the max
            left_p = np.max(boundary[boundary < 0])+mid_point
        else:
            left_p = np.min(boundary)+mid_point
        
        if boundary[boundary > 0].shape[0]:
            right_p = np.min(boundary[boundary > 0])+mid_point
        else:
            right_p = np.max(boundary)+mid_point
        
        mid_point = (left_p+right_p)//2
        row_idx = h-1-i*(step)

        yield (mid_point,row_idx)
    
def fitPath(mid_points):
    x_fit,y_fit = mid_points[:,1],mid_points[:,0]
    # x_fit,y_fit = x- x.mean(),y -y.mean()
    return np.polyfit(x_fit,y_fit,deg = 2)

def addObjects(objects,grid):
    #Should be
    for x,y,w,h in objects:
        grid[0:y+h,x:x+w] = 255
    return grid

def getpath(blue,yellow,objects,block_size,diag):
    i_left,i_right = np.nonzero(diag[-1])[0] 
    # print(i_left,i_right) 
    lanes = ([findStart(l,i_start,side,block_size) for l,side,i_start in zip([blue,yellow],['left','right'],(i_left,i_right)) ])
    starting_points = lanes[0][1]+lanes[1][1]

    grid = SetMasks(lanes,diag,starting_points)
    grid = addObjects(objects,grid)
    mid_points = np.array([ mid_point for mid_point in getMidPoint(grid,(lanes[0][1],lanes[1][1]) )])
    return mid_points,grid



if __name__ == '__main__':
    

    hsv_masks =  {
            'blue' : ( np.array([180//2, 63, 63]) , np.array([270//2, 255, 255])),
            'yellow':(np.array([40//2, 10, 63]),np.array([70//2, 255, 255]))
        }

    import time
    vid = cv2.VideoCapture('Birdseye.mp4')
    frames = 0
    t = 0
    block_size = (10,10)
    while(1):
        start = time.perf_counter()
        ret,img = vid.read()
        
        if ret:
            blue,yellow = detect_lane(img,hsv_masks)
            
            line,grid =  getCurve(blue,yellow,block_size)

            print(line)

            t += time.perf_counter()-start
            frames += 1
            cv2.imshow('path',grid)
            cv2.imshow('og',img)
            k = cv2.waitKey(200)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        else:
            break

    print(f'FPS : {frames/t}\n Second Between Frames: {t/frames}')