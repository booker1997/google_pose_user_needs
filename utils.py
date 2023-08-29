import random
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt
from ergonomics.reba import RebaScore
from reba_score_class import RebaScoreMIT

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

def calc_reba_lib(pose_array,verbose = False):
    rebaScore = RebaScore()

    body_params = rebaScore.get_body_angles_from_pose_right(pose_array)
    arms_params = rebaScore.get_arms_angles_from_pose_right(pose_array)

    rebaScore.set_body(body_params)
    score_a, partial_a = rebaScore.compute_score_a()

    rebaScore.set_arms(arms_params)
    score_b, partial_b = rebaScore.compute_score_b()

    score_c, caption = rebaScore.compute_score_c(score_a, score_b)
    danger_class = rebaScore.score_c_to_5_classes(score_c=score_c)
    if verbose:
        print("Score A: ", score_a, "Partial: ", partial_a)
        print("Score A: ", score_b, "Partial: ", partial_b)
        print("Score C: ", score_c, caption)

    return {'a':(score_a,partial_a),'b':(score_b,partial_b),'c':(score_c,caption),'class':danger_class,'caption':caption}


def calc_reba_custom(full_pose_dict,plot=False):
    a_results = []
    b_results = []
    c_results = []
    reba_class = []
    for i,frame in enumerate(full_pose_dict['neck']):
        frame_angle_dict = {}
        for key in full_pose_dict:
            frame_angle_dict[key] = full_pose_dict[key][i]
        reba_score = RebaScoreMIT(frame_angle_dict)
        score_a, partial_a = reba_score.compute_score_a()
        score_b, partial_b = reba_score.compute_score_b()
        score_c, caption = reba_score.compute_score_c(score_a,score_b)
        danger_class = reba_score.score_c_to_5_classes(score_c=score_c)
        a_results.append(score_a)
        b_results.append(score_b)
        c_results.append(score_c)
        reba_class.append(danger_class)
       
    # print(final_results_array_dict)
    if plot:
        plt.title('reba scores')
        plt.plot(range(len(a_results)),a_results)
        plt.plot(range(len(b_results)),b_results)
        plt.plot(range(len(c_results)),c_results)
        plt.legend(['a_score, [neck_score, trunk_score, leg_score]','b_score, [upper_arm_score, lower_arm_score, wrist_score]','c_score'])
        plt.show()

        plt.title('reba classes')
        plt.plot(range(len(reba_class)),reba_class)
        plt.yticks([0,1,2,3,4],['Negligible Risk',
                                'Low Risk',
                                'Medium Risk',
                                'High Risk',
                                'Very High Risk'
                                ])
        plt.show()
    return a_results,b_results,c_results,reba_class



def calc_and_plot_reba_w_lib(final_results_dict,plot=False):
    # print(final_results_dict)
    final_results_array_dict = {}
    for frame in range(len(final_results_dict['nose']['x'])):
        frame_pose_array = np.ones((len(final_results_dict),3))
        for i,key in enumerate(final_results_dict):
            data_dict = final_results_dict[key] 
            frame_pose_array[i] = np.array([data_dict['x'][frame],data_dict['y'][frame],data_dict['z'][frame]])
    
        final_results_array_dict[frame] = frame_pose_array

    # print(len(final_results_array_dict),frame_pose_array.shape,frame_pose_array)

    a_results = []
    b_results = []
    c_results = []
    reba_class = []
    # print(final_results_array_dict)
    for frame in final_results_array_dict:
        pose_array = final_results_array_dict[frame]
        results_dict = calc_reba_lib(pose_array)
        a_results.append(results_dict['a'][0])
        b_results.append(results_dict['b'][0])
        c_results.append(results_dict['c'][0])
        reba_class.append(results_dict['class'])
    if plot:
        plt.title('reba scores')
        plt.plot(range(len(a_results)),a_results)
        plt.plot(range(len(b_results)),b_results)
        plt.plot(range(len(c_results)),c_results)
        plt.legend(['a_score, [neck_score, trunk_score, leg_score]','b_score, [upper_arm_score, lower_arm_score, wrist_score]','c_score'])
        plt.show()

        plt.title('reba classes')
        plt.plot(range(len(reba_class)),reba_class)
        plt.yticks([0,1,2,3,4],['Negligible Risk',
                                'Low Risk',
                                'Medium Risk',
                                'High Risk',
                                'Very High Risk'
                                ])
        plt.show()
    return a_results,b_results,c_results,reba_class