from Calibrate.calibrate import calibrate 
from Classification.classification import turnDirection
from ObjectDetection.object_detection import detect_object
from LaneDetection.lane_detection import detect_lane


#Executes all the camera stuff and returns a dictionary for the state to control

def GetCameraData(img,turn_net,obj_net,camera_settings):

    hsv_masks = camera_settings['hsv_masks']
    img = calibrate(img)
    turn = turnDirection(img,turn_net)
    obstacle =detect_object(img,obj_net )
    b_lane,y_lane = detect_lane(img,hsv_masks)

    return {'turn':turn,
            'obstacle': obstacle,
            'blue_lane': b_lane,
            'yellow_lane':y_lane
             }