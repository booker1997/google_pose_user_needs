import random
from utils import *
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt
from reba_score_class import RebaScoreMIT
import pandas as pd
from scipy.signal import find_peaks


def reba_video_analyzer(video_file_path=None,test=False,frontview=True,show_plots=False,camera_frames_per_second = 30,create_csv_from_data = True):

    if test:
        # input_video_name = 'booker_trimmed.mp4'
        # input_video_name = 'booker_test.MOV'
        # input_video_name = 'reba_test_videos/low_score/low_score.MOV'
        # input_video_name = 'reba_test_videos/medium_score/medium_score.MOV'
        input_video_name = 'reba_test_videos/high_score/v2/high_score_v2.MOV'
        give_processing_updates_time = 5
    else:
        input_video_name = video_file_path
        give_processing_updates_time = 5

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # print(sys.argv)
    cap = cv2.VideoCapture(input_video_name)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    #     '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
    # inflnm, inflext = inputflnm.split('.')
    print('working on video...')
    out_filename = input_video_name[:-4] +'_annotated.avi' #f'{outdir}{inflnm}_annotated.{inflext}'
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


    final_results_l_shoulder = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_hip = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_knee = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_knee = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_shoulder = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_hip = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_nose = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_wrist = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_wrist = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_elbow = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_elbow = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_ankle = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_ankle = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_index = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_index = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_r_pinky = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    final_results_l_pinky = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[],'z_frame':[]}
    # final_results_head = {'x':[],'y':[],'z':[],'visibility':[],'x_frame':[],'y_frame':[]}

    frame_ct = 0
    second_processed_count = 0
    print_update = True
    results_storage = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        if frame_ct == 0 or frame_ct == 45: # this plots the pose estimation in the 3d coordinate system
            results_storage.append(results)
            # print('Nose world landmark:'),
            # print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
            # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        frame_ct += 1
        if frame_ct%camera_frames_per_second == 0:
            second_processed_count += 1
            print_update = True
        if second_processed_count %give_processing_updates_time == 0 and print_update :
            print(f'{second_processed_count} seconds of video processed...') 
            print_update = False
        
        try: 
            nose = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
    
            l_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            l_shoulder_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_shoulder_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            l_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            l_hip_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            r_hip_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            
            l_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            l_knee_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            r_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_knee_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            l_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            l_ankle_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            r_ankle_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            l_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            l_elbow_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_elbow_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

            l_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            l_wrist_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            r_wrist_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            l_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            l_index_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            r_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
            r_index_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

            l_pinky = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY]
            l_pinky_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY]
            r_pinky= results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
            r_pinky_frame = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
            
            joint_names = (l_shoulder,r_shoulder,l_hip,r_hip,l_knee,r_knee,l_ankle,r_ankle,nose,l_elbow,r_elbow,l_wrist,r_wrist,l_index,r_index,l_pinky,r_pinky)
            joint_frame_names = [l_shoulder_frame,r_shoulder_frame,l_hip_frame,r_hip_frame,l_knee_frame,r_knee_frame,l_ankle_frame,r_ankle_frame,nose_frame
                                ,l_elbow_frame,r_elbow_frame,l_wrist_frame,r_wrist_frame,l_index_frame,r_index_frame,l_pinky_frame,r_pinky_frame]
            final_results_list = [final_results_l_shoulder,final_results_r_shoulder,final_results_l_hip,final_results_r_hip,final_results_l_knee,final_results_r_knee,
                                    final_results_l_ankle,final_results_r_ankle,final_results_nose,final_results_l_elbow,final_results_r_elbow,final_results_l_wrist,
                                        final_results_r_wrist,final_results_l_index,final_results_r_index,final_results_l_pinky,final_results_r_pinky]
            final_dict_keys = ['x','y','z','visibility','x_frame','y_frame']
   
            for i_joint,joint in enumerate(joint_names):
                final_results_list[i_joint]['x'].append(joint.x)
                final_results_list[i_joint]['y'].append(joint.y)
                final_results_list[i_joint]['z'].append(joint.z)
                final_results_list[i_joint]['x_frame'].append(round(joint_frame_names[i_joint].x*frame_width))
                final_results_list[i_joint]['y_frame'].append(round(joint_frame_names[i_joint].y*frame_height))
                final_results_list[i_joint]['z_frame'].append(joint_frame_names[i_joint].z)
                final_results_list[i_joint]['visibility'].append(round(joint.visibility,2))
           

            
        # Plot pose world landmarks.
            # if random.random() < 0.05:
            #     plot = mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        except:
            pass

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(results.pose_landmarks)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(image)
    pose.close()
    cap.release()
    out.release()
    final_results_dict = {'nose':final_results_nose,
                                'l_shoulder':final_results_l_shoulder,'l_elbow':final_results_l_elbow,'l_wrist':final_results_l_wrist,
                                'r_shoulder':final_results_r_shoulder,'r_elbow':final_results_r_elbow,'r_wrist':final_results_r_wrist,
                                'l_hip':final_results_l_hip,'l_knee':final_results_l_knee,'l_ankle':final_results_l_ankle,
                                'r_hip':final_results_r_hip,'r_knee':final_results_r_knee,'r_ankle':final_results_r_ankle}
                                


    print('CALCULATING ANGLES...',f'total frames = {frame_ct}')


    mid_hip = calc_midpoint_vec(final_results_l_hip,final_results_r_hip)
    mid_shoulder = calc_midpoint_vec(final_results_l_shoulder,final_results_r_shoulder)
    mid_knee = calc_midpoint_vec(final_results_l_knee,final_results_r_knee)
    mid_r_fingers = calc_midpoint_vec(final_results_r_index,final_results_r_pinky)
    mid_l_fingers = calc_midpoint_vec(final_results_l_index,final_results_l_pinky)

    hip_angle_ref_point = np.array([0,-5,0])


    l_shoulder_hip_ref_points = find_point_on_line(final_results_l_hip,final_results_r_hip,final_results_l_shoulder)
    r_shoulder_hip_ref_points = find_point_on_line(final_results_l_hip,final_results_r_hip,final_results_r_shoulder)
    l_upper_arm_frontal_angles,l_upper_arm_sagital_angles,l_projected_ref_points = get_sagital_and_frontal_angles(final_results_r_shoulder,final_results_l_hip,final_results_l_shoulder,final_results_l_elbow,hip_ref_points = l_shoulder_hip_ref_points)
    r_upper_arm_frontal_angles,r_upper_arm_sagital_angles,r_projected_ref_points = get_sagital_and_frontal_angles(final_results_l_shoulder,final_results_r_hip,final_results_r_shoulder,final_results_r_elbow,hip_ref_points = r_shoulder_hip_ref_points)

    # For testing transformations
    # test_frame = 181
    # print('l_shoulder_sagital_angle',l_upper_arm_sagital_angles[test_frame])
    # print('l_shoulder_frontal_angle',l_upper_arm_frontal_angles[test_frame])
    # print('r_shoulder_sagital_angle',r_upper_arm_sagital_angles[test_frame])
    # print('r_shoulder_frontal_angle',r_upper_arm_frontal_angles[test_frame])
    # plot_ref_points_w_body(final_results_dict,l_projected_ref_points+r_projected_ref_points,test_frame)
    if show_plots:
        plt.plot(range(len(l_upper_arm_sagital_angles)),l_upper_arm_sagital_angles)
        plt.plot(range(len(l_upper_arm_frontal_angles)),l_upper_arm_frontal_angles)
        plt.xlabel('frame')
        plt.ylabel('angle (deg)')
        plt.legend(['l_upper_arm_sagital_angles','l_upper_arm_frontal_angles'])
        plt.show()
        plt.plot(range(len(r_upper_arm_sagital_angles)),r_upper_arm_sagital_angles)
        plt.plot(range(len(r_upper_arm_frontal_angles)),r_upper_arm_frontal_angles)
        plt.xlabel('frame')
        plt.ylabel('angle (deg)')
        plt.legend(['r_upper_arm_sagital_angles','r_upper_arm_frontal_angles'])
        plt.show()
    #mp_drawing.plot_landmarks(results_storage[1].pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # l_shoulder_ref_point = np.array([on_plane_angle_mid_joint['x'],np.ones(len(final_results_l_shoulder['x']))*5,final_results_l_shoulder['z']])
    # l_shoulder_ref_point = np.array([final_results_l_shoulder['x'],np.ones(len(final_results_l_shoulder['x']))*5,final_results_l_shoulder['z']])

    v1_l_hip = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_l_hip['x']),
    np.array(final_results_l_shoulder['z'])-np.array(final_results_l_hip['z'])])

    v2_l_hip =  np.array([np.array(final_results_l_knee['x'])-np.array(final_results_l_hip['x']),
    np.array(final_results_l_knee['y'])-np.array(final_results_l_hip['y']),
    np.array(final_results_l_knee['z'])-np.array(final_results_l_hip['z'])])

    # v1_mid_hip = np.array([np.array(mid_shoulder['x'])-np.array(mid_hip['x']),
    # np.array(mid_shoulder['y'])-np.array(mid_hip['y']),
    # np.array(mid_shoulder['z'])-np.array(mid_hip['z'])])

    # v2_mid_hip =  np.array([np.array(mid_knee['x'])-np.array(mid_hip['x']),
    # np.array(mid_knee['y'])-np.array(mid_hip['y']),
    # np.array(mid_knee['z'])-np.array(mid_hip['z'])])

    v1_mid_hip = np.array([np.array(mid_shoulder['x'])-np.array(mid_hip['x']),
    np.array(mid_shoulder['y'])-np.array(mid_hip['y']),
    np.array(mid_shoulder['z'])-np.array(mid_hip['z'])])

    v2_mid_hip =  np.array([hip_angle_ref_point[0]-np.array(mid_hip['x']),
    hip_angle_ref_point[1]-np.array(mid_hip['y']),
    hip_angle_ref_point[2]-np.array(mid_hip['z'])])

    v1_l_elbow = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_l_elbow['x']),
    np.array(final_results_l_shoulder['y'])-np.array(final_results_l_elbow['y']),
    np.array(final_results_l_shoulder['z'])-np.array(final_results_l_elbow['z'])])

    v2_l_elbow =  np.array([np.array(final_results_l_wrist['x'])-np.array(final_results_l_elbow['x']),
    np.array(final_results_l_wrist['y'])-np.array(final_results_l_elbow['y']),
    np.array(final_results_l_wrist['z'])-np.array(final_results_l_elbow['z'])])

    v1_r_elbow = np.array([np.array(final_results_r_shoulder['x'])-np.array(final_results_r_elbow['x']),
    np.array(final_results_r_shoulder['y'])-np.array(final_results_r_elbow['y']),
    np.array(final_results_r_shoulder['z'])-np.array(final_results_r_elbow['z'])])

    v2_r_elbow =  np.array([np.array(final_results_r_wrist['x'])-np.array(final_results_r_elbow['x']),
    np.array(final_results_r_wrist['y'])-np.array(final_results_r_elbow['y']),
    np.array(final_results_r_wrist['z'])-np.array(final_results_r_elbow['z'])])


    v1_l_knee = np.array([np.array(final_results_r_hip['x'])-np.array(final_results_l_knee['x']),
    np.array(final_results_r_hip['y'])-np.array(final_results_l_knee['y']),
    np.array(final_results_r_hip['z'])-np.array(final_results_l_knee['z'])])

    v2_l_knee =  np.array([np.array(final_results_l_ankle['x'])-np.array(final_results_l_knee['x']),
    np.array(final_results_l_ankle['y'])-np.array(final_results_l_knee['y']),
    np.array(final_results_l_ankle['z'])-np.array(final_results_l_knee['z'])])

    v1_r_knee = np.array([np.array(final_results_r_hip['x'])-np.array(final_results_r_knee['x']),
    np.array(final_results_r_hip['y'])-np.array(final_results_r_knee['y']),
    np.array(final_results_r_hip['z'])-np.array(final_results_r_knee['z'])])

    v2_r_knee =  np.array([np.array(final_results_r_ankle['x'])-np.array(final_results_r_knee['x']),
    np.array(final_results_r_ankle['y'])-np.array(final_results_r_knee['y']),
    np.array(final_results_r_ankle['z'])-np.array(final_results_r_knee['z'])])

    v1_neck = np.array([np.array(final_results_nose['x'])-np.array(mid_shoulder['x']),
    np.array(final_results_nose['y'])-np.array(mid_shoulder['y']),
    np.array(final_results_nose['z'])-np.array(mid_shoulder['z'])])

    v2_neck =  np.array([np.array(mid_hip['x'])-np.array(mid_shoulder['x']),
    np.array(mid_hip['y'])-np.array(mid_shoulder['y']),
    np.array(mid_hip['z'])-np.array(mid_shoulder['z'])])

    v1_l_wrist = np.array([np.array(final_results_l_elbow['x'])-np.array(final_results_l_wrist['x']),
    np.array(final_results_l_elbow['y'])-np.array(final_results_l_wrist['y']),
    np.array(final_results_l_elbow['z'])-np.array(final_results_l_wrist['z'])])

    v2_l_wrist =  np.array([np.array(mid_l_fingers['x'])-np.array(final_results_l_wrist['x']),
    np.array(mid_l_fingers['y'])-np.array(final_results_l_wrist['y']),
    np.array(mid_l_fingers['z'])-np.array(final_results_l_wrist['z'])])

    v1_r_wrist = np.array([np.array(final_results_r_elbow['x'])-np.array(final_results_r_wrist['x']),
    np.array(final_results_r_elbow['y'])-np.array(final_results_r_wrist['y']),
    np.array(final_results_r_elbow['z'])-np.array(final_results_r_wrist['z'])])

    v2_r_wrist =  np.array([np.array(mid_r_fingers['x'])-np.array(final_results_r_wrist['x']),
    np.array(mid_r_fingers['y'])-np.array(final_results_r_wrist['y']),
    np.array(mid_r_fingers['z'])-np.array(final_results_r_wrist['z'])])

    # Calculate angles
    angle_deg_neck = get_joint_angles(v1_neck,v2_neck)
    angle_deg_neck = 180-angle_deg_neck
    angle_deg_neck = angle_deg_neck-(np.ones_like(angle_deg_neck)*angle_deg_neck[0])

    angle_deg_mid_hip = get_joint_angles(v1_mid_hip,v2_mid_hip)
    mid_hip_visibility_score_avg = (np.array(final_results_l_hip['visibility']) + np.array(final_results_r_hip['visibility'])  + np.array(final_results_l_shoulder['visibility']) + np.array(final_results_r_shoulder['visibility']) )/4
    angle_deg_l_knee = get_joint_angles(v1_l_knee,v2_l_knee)
    angle_deg_r_knee = get_joint_angles(v1_r_knee,v2_r_knee)
    angle_deg_l_knee = 180-angle_deg_l_knee
    angle_deg_r_knee = 180-angle_deg_r_knee

    angle_deg_l_elbow = get_joint_angles(v1_l_elbow,v2_l_elbow)
    angle_deg_r_elbow = get_joint_angles(v1_r_elbow,v2_r_elbow)
    angle_deg_r_elbow = angle_deg_r_elbow
    angle_deg_l_elbow = angle_deg_l_elbow

    angle_deg_l_wrist = get_joint_angles(v1_l_wrist,v2_l_wrist)
    angle_deg_r_wrist = get_joint_angles(v1_r_wrist,v2_r_wrist)
    angle_deg_l_wrist = 180-angle_deg_l_wrist-10
    angle_deg_r_wrist = 180-angle_deg_r_wrist-10

    reba_angles = {'l_upper_arm':(l_upper_arm_sagital_angles,l_upper_arm_frontal_angles),'r_upper_arm':(r_upper_arm_sagital_angles,r_upper_arm_frontal_angles),'l_lower_arm':angle_deg_l_elbow,'r_lower_arm':angle_deg_r_elbow,
                        'l_wrist':angle_deg_l_wrist,'r_wrist':angle_deg_r_wrist,
                        'l_knee':angle_deg_l_knee,'r_knee':angle_deg_r_knee,'neck':angle_deg_neck,'trunk_angle':angle_deg_mid_hip}
    reba_angles_dataframe = {'l_upper_arm_sag':l_upper_arm_sagital_angles,'l_upper_arm_frontal':l_upper_arm_frontal_angles,'r_upper_arm_sag':r_upper_arm_sagital_angles,'r_upper_arm_frontal':r_upper_arm_frontal_angles,'l_lower_arm':angle_deg_l_elbow,'r_lower_arm':angle_deg_r_elbow,
                        'l_wrist':angle_deg_l_wrist,'r_wrist':angle_deg_r_wrist,
                        'l_knee':angle_deg_l_knee,'r_knee':angle_deg_r_knee,'neck':angle_deg_neck,'trunk_angle':angle_deg_mid_hip}

    # Calculate Reba score

    a_results,b_results,c_results,reba_class,upper_arm_scores,lower_arm_scores,wrist_scores,neck_scores,trunk_scores,leg_scores =calc_reba_custom(reba_angles,plot=show_plots)
    

    reba_angles_dataframe['upper_arm_score'] = upper_arm_scores
    reba_angles_dataframe['lower_arm_score'] = lower_arm_scores
    reba_angles_dataframe['wrist_score'] = wrist_scores
    reba_angles_dataframe['neck_score'] = neck_scores
    reba_angles_dataframe['trunk_score'] = trunk_scores
    reba_angles_dataframe['leg_score'] = leg_scores

    reba_angles_dataframe['a_score'] = a_results
    reba_angles_dataframe['b_score'] = b_results
    reba_angles_dataframe['c_score'] = c_results

    if create_csv_from_data:
        dataframe = pd.DataFrame(data=reba_angles_dataframe)

        dataframe.to_csv('reba_test_videos/data/reba_data.csv')

    #Post process and create peaks dataframe

    max_reba_c = max(c_results)
    find_peaks_above = 3
    peak_i = find_peaks(c_results,find_peaks_above)[0]
    peak_mags = np.array(c_results)[peak_i]
 
    
    frames_to_include_around_peaks = 10
    peaks_dict = {}
    frames_by_peaks = np.array([])
    for i in peak_i:
        if i>frames_to_include_around_peaks:
            start = i-frames_to_include_around_peaks
            end = i+frames_to_include_around_peaks
            peaks_dict[i] = np.arange(start,end)
            
        elif i > round(frames_to_include_around_peaks/2):
            start = i-round(frames_to_include_around_peaks/2)
            end = i+round(frames_to_include_around_peaks/2)
            peaks_dict[i] = np.arange(start,end)
    peaks_dataframe = {}
    reba_angles_dataframe_keys_reversed = list(reba_angles_dataframe.keys())
    reba_angles_dataframe_keys_reversed.reverse()
    create_init_list = True
    create_init_coords = True
    create_init_list_angles = True
    for i,peak_i in enumerate(peaks_dict):
        peak_val = peak_mags[i]
        if peak_val == 1:
            max_risk_level = 'No Risk'
        elif 2 <= peak_val <= 3:
            max_risk_level = 'Low Risk'
        elif 4 <= peak_val <= 7:
            max_risk_level = 'Medium Risk'
        elif 8 <= peak_val <= 10:
            max_risk_level = 'High Risk'
        else:
            max_risk_level = 'Very High Risk'
        for frame in peaks_dict[peak_i]:
            if create_init_list:
                peaks_dataframe['frame_id'] = [frame]
                peaks_dataframe['frame_of_max_peak'] = [peak_i]
                peaks_dataframe['frame_c_score'] = [reba_angles_dataframe['c_score'][frame]]
                peaks_dataframe['peak_risk_level'] = [max_risk_level]
                create_init_list = False
            else:
                peaks_dataframe['frame_id'].append(frame)
                peaks_dataframe['frame_of_max_peak'].append(peak_i)
                peaks_dataframe['frame_c_score'].append(reba_angles_dataframe['c_score'][frame])
                peaks_dataframe['peak_risk_level'].append(max_risk_level)
            for joint_name in final_results_dict:
                if create_init_coords:
                    peaks_dataframe[joint_name + '_frame_x'] = [final_results_dict[joint_name]['x_frame'][frame]]
                    peaks_dataframe[joint_name + '_frame_y'] = [final_results_dict[joint_name]['y_frame'][frame]]
                    peaks_dataframe[joint_name + '_frame_z_norm'] = [final_results_dict[joint_name]['z_frame'][frame]]
                    peaks_dataframe[joint_name + '_visibility'] = [final_results_dict[joint_name]['visibility'][frame]]
                    
                else:
                    peaks_dataframe[joint_name + '_frame_x'].append(final_results_dict[joint_name]['x_frame'][frame])
                    peaks_dataframe[joint_name + '_frame_y'].append(final_results_dict[joint_name]['y_frame'][frame])
                    peaks_dataframe[joint_name + '_frame_z_norm'].append(final_results_dict[joint_name]['z_frame'][frame])
                    peaks_dataframe[joint_name + '_visibility'].append(final_results_dict[joint_name]['visibility'][frame])
            create_init_coords = False
                    
            for data_key in reba_angles_dataframe_keys_reversed:
                if create_init_list_angles:
                    peaks_dataframe[data_key] = [reba_angles_dataframe[data_key][frame]]
                else:
                    peaks_dataframe[data_key].append(reba_angles_dataframe[data_key][frame])
            create_init_list_angles = False
    dataframe_output = pd.DataFrame(data=peaks_dataframe)

    dataframe_output.to_csv('reba_test_videos/data/peaks_output_by_frame.csv')

    


    #Make subplots for visualization 
    if show_plots:
        plt.figure(1)
        plt.subplot(2,7,1)
        plt.plot(range(len(angle_deg_mid_hip)),angle_deg_mid_hip,color='r', label='angle deg')
        plt.title('REBA Trunk angles')

        plt.subplot(2,7,2)
        plt.plot(range(len(mid_hip_visibility_score_avg)),mid_hip_visibility_score_avg)
        plt.title('REBA Trunk angles visibility')

        plt.subplot(2,7,3)
        plt.plot(range(len(angle_deg_l_knee)),angle_deg_l_knee)
        plt.title('REBA L Knee angles')
        plt.subplot(2,7,4)
        plt.plot(range(len(angle_deg_r_knee)),angle_deg_r_knee)
        plt.title('REBA R Knee angles')

        plt.subplot(2,7,5)
        plt.plot(range(len(l_upper_arm_sagital_angles)),l_upper_arm_sagital_angles)
        plt.title('REBA L Shoulder sag angles')
        plt.subplot(2,7,6)
        plt.plot(range(len(l_upper_arm_frontal_angles)),l_upper_arm_frontal_angles)
        plt.title('REBA L Shoulder frontal angles')

        plt.subplot(2,7,7)
        plt.plot(range(len(r_upper_arm_sagital_angles)),r_upper_arm_sagital_angles)
        plt.title('REBA R Shoulder Sag angles')
        plt.subplot(2,7,8)
        plt.plot(range(len(r_upper_arm_frontal_angles)),r_upper_arm_frontal_angles)
        plt.title('REBA R Shoulder Frontal angles')

        plt.subplot(2,7,9)
        plt.plot(range(len(angle_deg_l_elbow)),angle_deg_l_elbow)
        plt.title('REBA L elbow angles')
        plt.subplot(2,7,10)
        plt.plot(range(len(angle_deg_r_elbow)),angle_deg_r_elbow)
        plt.title('REBA R elbow angles')

        plt.subplot(2,7,11)
        plt.plot(range(len(angle_deg_neck)),angle_deg_neck)
        plt.title('REBA neck angles')

        plt.subplot(2,7,12)
        plt.plot(range(len(angle_deg_l_wrist)),angle_deg_l_wrist)
        plt.title('REBA l wrist angles')

        plt.subplot(2,7,13)
        plt.plot(range(len(angle_deg_r_wrist)),angle_deg_r_wrist)
        plt.title('REBA r wrist angles')

        plt.show()


    return peaks_dataframe


