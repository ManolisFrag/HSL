import numpy as np
import json
import os
from math import *
from math import pi as pii
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

def scale_x(x_array, min_array_x, max_array_x):
    OldMax = max_array_x
    OldMin = min_array_x
    NewMax = 1
    NewMin = -1
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    Newx = []
    for OldValuex in x_array:
        NewValue = (((OldValuex - OldMin) * NewRange) / OldRange) + NewMin
        Newx.append(NewValue)
    return Newx

def scale_y(y_array, min_array_y, max_array_y):    
    Newy = []    
    OldMax = max_array_y
    OldMin = min_array_y
    NewMax = 1
    NewMin = -1
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    for OldValuey in y_array:
        NewValuey = (((OldValuey - OldMin) * NewRange) / OldRange) + NewMin
        Newy.append(NewValuey)
    return Newy

def getAngle(A,B):
    
    deltaY = abs(B[1] - A[1]);
    deltaX = abs(B[0] - A[0]);

    angleInDegrees = atan2(deltaY, deltaX) * 180 / pi
    print("AngleINDegrees raw is: ", angleInDegrees)
    if(B[1] > A[1]):

        if(B[0] < A[0]):
            print("case 1")
            angleInDegrees += 180;
        else: 
            print("case 2")
            angleInDegrees += 270;

    elif (B[0] < A[0]):
        print("case 3")
        angleInDegrees += 90;

    return(angleInDegrees)


def angle_c(x1,y1,x2,y2):
    if x1==x2:
        if (y2 > y1):
            result = 0.5 * pii
        else:
            result = 1.5 * pii
    result = atan((y2-y1)/(x2-x1))
    if (x2 < x1):
        result = result + pii
    if (result < 0):
        result = result + 2*pii
    result = result * 180/pii
    return(result)
        
            
    
def get_arm_angle_from_json(json_file_path):
    with open(json_file_path) as f:
        loaded_json = json.load(f)
        raw_coords = loaded_json["people"][0]["pose_keypoints_2d"]
        #remove confidence values
        raw_coords = np.delete(raw_coords, np.arange(2, len(raw_coords), 3))
        raw_coords = np.reshape(raw_coords, (25,2))

        raw_coords = np.array(raw_coords)
        arm_coords = np.array([raw_coords[3],raw_coords[4]])
        #print(arm_coords[0][0],arm_coords[0][1], arm_coords[1][0], arm_coords[1][0])
       
        return(angle_c(arm_coords[0][0],arm_coords[0][1], arm_coords[1][0], arm_coords[1][1]))
    
def get_left_fingers_from_json(json_file_path):
    """
    function to open a json file and return the x,y positions of the fingers of the left hand.
    Provide the path to the json file
    """
    with open(json_file_path) as f:
        loaded_json = json.load(f)
        person = loaded_json["people"]
        if person:
            raw_coords = loaded_json["people"][0]["hand_left_keypoints_2d"]
            #remove confidence values
    #         raw_coords = np.delete(raw_coords, np.arange(2, len(raw_coords), 3))
            raw_coords = np.reshape(raw_coords, (21,3))

            raw_coords = np.array(raw_coords)

            scaled_x = scale_x(raw_coords[:,[0]], min(raw_coords[:,[0]]), max(raw_coords[:,[0]]))
            scaled_y = scale_y(raw_coords[:,[1]], min(raw_coords[:,[1]]), max(raw_coords[:,[1]]))
            confidence_left = raw_coords[:,[2]]
            return(scaled_x,scaled_y,confidence_left) 
        
## rotate hand
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

    

class handshape(object):
    
    def __init__(self, json_file_path):
        self._json_file_path = json_file_path
    
    @property
    def get_right_fingers_from_json(self):
        """
        function to open a json file and return the x,y positions of the fingers of the left hand.
        Provide the path to the json file
        """
        with open(self._json_file_path) as f:
            loaded_json = json.load(f)
            person = loaded_json["people"]
            if person:
                raw_coords = loaded_json["people"][0]["hand_right_keypoints_2d"]

                raw_coords = np.reshape(raw_coords, (21,3))

                raw_coords = np.array(raw_coords)

                scaled_x = scale_x(raw_coords[:,[0]], min(raw_coords[:,[0]]), max(raw_coords[:,[0]]))
                scaled_y = scale_y(raw_coords[:,[1]], min(raw_coords[:,[1]]), max(raw_coords[:,[1]]))
                confidence_right = raw_coords[:,[2]]

                #get angle of arm from json
                rotation = get_arm_angle_from_json(self._json_file_path)

                #debug: plot scalled coordinates
    #             plt.scatter(scaled_x, scaled_y, s=(20 * confidence_right)**2, alpha = 0.7)
    #             plt.show()

                pi=22/7
                rotated_hand_coords = []
                for i in range (len(scaled_x)):
                    rotated_hand_coords.append(rotate((scaled_x[i][0],scaled_y[i][0]), (0,0), radians(rotation*(pi/180))))
                rotated_hand_coords = np.array(rotated_hand_coords)
                return(rotated_hand_coords, confidence_right)
                #return([scaled_x,scaled_y,confidence_right]) 
            else:
                pass
