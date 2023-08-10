import random
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt

def get_joint_angles(v1,v2):
    v1mag = np.sqrt(v1[0,:] * v1[0,:] + v1[1,:] * v1[1,:] + v1[2,:]  * v1[2,:] )
    v1norm = [v1[0,:] / v1mag, v1[1,:] / v1mag, v1[2,:] / v1mag]

    v2mag = np.sqrt(v2[0,:] * v2[0,:] + v2[1,:] * v2[1,:] + v2[2,:]  * v2[2,:] )
    v2norm = [v2[0,:] / v2mag, v2[1,:] / v2mag, v2[2,:] / v2mag]

    res = (v1norm[0] * v2norm[0]) + (v1norm[1] * v2norm[1]) + (v1norm[2] * v2norm[2])

    angle_rad = np.arccos(res)
    angle_deg = np.rad2deg(angle_rad)

    return (angle_deg,angle_rad)

def calc_midpoint_vec(left,right):
    
    x = (np.array(left['x']) + np.array(right['x']))/2
    y = (np.array(left['y']) + np.array(right['y']))/2
    z = (np.array(left['z']) + np.array(right['z']))/2

    v_mid = {'x':x,'y':y,'z':z}

    return v_mid

def calc_distance_between_nodes(node_1,node_2):
    norms = np.sqrt((np.array(node_1['x'])-np.array(node_2['x']))**(2) + (np.array(node_1['y'])-np.array(node_2['y']))**(2) +
                                            (np.array(node_1['z'])-np.array(node_2['z']))**(2) )
    return norms