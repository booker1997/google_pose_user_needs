import random
from utils import *
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt
from ergonomics.reba import RebaScore
from reba_score_class import RebaScoreMIT
import pandas as pd


test = True
front_view = True

if test:
    # input_video_name = 'booker_trimmed.mp4'
    # input_video_name = 'booker_test.MOV'
    # input_video_name = 'reba_test_videos/low_score/low_score.MOV'
    # input_video_name = 'reba_test_videos/medium_score/medium_score.MOV'
    input_video_name = 'reba_test_videos/high_score/v2/high_score_v2.MOV'
    give_processing_updates_time = 5
else:
    if front_view:
        print('Front view selected!')
        input_video_name = 'scan_video2.avi'
        give_processing_updates_time = 15
    else:
        print('Side view selected!')
        input_video_name = 'scan_video1.avi'
        give_processing_updates_time = 15

camera_frames_per_second = 30


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


final_results_l_shoulder = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_hip = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_knee = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_knee = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_shoulder = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_hip = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_nose = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_wrist = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_wrist = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_elbow = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_elbow = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_ankle = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_ankle = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_index = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_index = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_r_pinky = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_l_pinky = {'x':[],'y':[],'z':[],'visibility':[]}
final_results_head = {'x':[],'y':[],'z':[],'visibility':[]}

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
        print(nose_frame.x*frame_width,frame_width,frame_height)
 
        l_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder= results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        l_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        l_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        l_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        l_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        l_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        l_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        r_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

        l_pinky = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY]
        r_pinky= results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
        
        final_results_l_shoulder['x'].append(l_shoulder.x)
        final_results_l_shoulder['y'].append(l_shoulder.y)
        final_results_l_shoulder['z'].append(l_shoulder.z)
        final_results_l_shoulder['visibility'].append(round(l_shoulder.visibility,2))

        final_results_r_shoulder['x'].append(r_shoulder.x)
        final_results_r_shoulder['y'].append(r_shoulder.y)
        final_results_r_shoulder['z'].append(r_shoulder.z)
        final_results_r_shoulder['visibility'].append(round(r_shoulder.visibility,2))

        final_results_l_hip['x'].append(l_hip.x)
        final_results_l_hip['y'].append(l_hip.y)
        final_results_l_hip['z'].append(l_hip.z)
        final_results_l_hip['visibility'].append(round(l_hip.visibility,2))

        final_results_r_hip['x'].append(r_hip.x)
        final_results_r_hip['y'].append(r_hip.y)
        final_results_r_hip['z'].append(r_hip.z)
        final_results_r_hip['visibility'].append(round(r_hip.visibility,2))

        final_results_l_knee['x'].append(l_knee.x)
        final_results_l_knee['y'].append(l_knee.y)
        final_results_l_knee['z'].append(l_knee.z)
        final_results_l_knee['visibility'].append(round(l_knee.visibility,2))

        final_results_r_knee['x'].append(r_knee.x)
        final_results_r_knee['y'].append(r_knee.y)
        final_results_r_knee['z'].append(r_knee.z)
        final_results_r_knee['visibility'].append(round(r_knee.visibility,2))

        final_results_l_ankle['x'].append(l_ankle.x)
        final_results_l_ankle['y'].append(l_ankle.y)
        final_results_l_ankle['z'].append(l_ankle.z)
        final_results_l_ankle['visibility'].append(round(l_ankle.visibility,2))

        final_results_r_ankle['x'].append(r_ankle.x)
        final_results_r_ankle['y'].append(r_ankle.y)
        final_results_r_ankle['z'].append(r_ankle.z)
        final_results_r_ankle['visibility'].append(round(r_ankle.visibility,2))

        final_results_nose['x'].append(nose.x)
        final_results_nose['y'].append(nose.y)
        final_results_nose['z'].append(nose.z)
        final_results_nose['visibility'].append(round(nose.visibility,2))

        final_results_l_elbow['x'].append(l_elbow.x)
        final_results_l_elbow['y'].append(l_elbow.y)
        final_results_l_elbow['z'].append(l_elbow.z)
        final_results_l_elbow['visibility'].append(l_elbow.visibility)

        final_results_r_elbow['x'].append(r_elbow.x)
        final_results_r_elbow['y'].append(r_elbow.y)
        final_results_r_elbow['z'].append(r_elbow.z)
        final_results_r_elbow['visibility'].append(r_elbow.visibility)

        final_results_l_wrist['x'].append(l_wrist.x)
        final_results_l_wrist['y'].append(l_wrist.y)
        final_results_l_wrist['z'].append(l_wrist.z)
        final_results_l_wrist['visibility'].append(round(l_wrist.visibility,2))

        final_results_r_wrist['x'].append(r_wrist.x)
        final_results_r_wrist['y'].append(r_wrist.y)
        final_results_r_wrist['z'].append(r_wrist.z)
        final_results_r_wrist['visibility'].append(round(r_wrist.visibility,2))

        final_results_l_index['x'].append(l_index.x)
        final_results_l_index['y'].append(l_index.y)
        final_results_l_index['z'].append(l_index.z)
        final_results_l_index['visibility'].append(round(l_index.visibility,2))

        final_results_r_index['x'].append(r_index.x)
        final_results_r_index['y'].append(r_index.y)
        final_results_r_index['z'].append(r_index.z)
        final_results_r_index['visibility'].append(round(r_index.visibility,2))

        final_results_l_pinky['x'].append(l_pinky.x)
        final_results_l_pinky['y'].append(l_pinky.y)
        final_results_l_pinky['z'].append(l_pinky.z)
        final_results_l_pinky['visibility'].append(round(l_pinky.visibility,2))

        final_results_r_pinky['x'].append(r_pinky.x)
        final_results_r_pinky['y'].append(r_pinky.y)
        final_results_r_pinky['z'].append(r_pinky.z)
        final_results_r_pinky['visibility'].append(round(r_pinky.visibility,2))

        # final_results_dict = {'nose':final_results_nose,
        #                         'l_shoulder':final_results_l_shoulder,'r_shoulder':final_results_r_shoulder,
        #                         'l_elbow':final_results_l_elbow,'r_elbow':final_results_r_elbow,
        #                         'l_wrist':final_results_l_wrist,'r_wrist':final_results_r_wrist,
        #                         'l_hip':final_results_l_hip,'r_hip':final_results_r_hip,
        #                         'l_knee':final_results_l_knee,'r_knee':final_results_r_knee,
        #                         'l_ankle':final_results_l_ankle,'r_ankle':final_results_r_ankle}

        
        # for key in final_results_dict:
        #     final_result = final_results_dict[key]
        #     final_result['x'] = np.array(final_result['x'])
        #     final_result['y'] = np.array(final_result['y'])
        #     final_result['z'] = np.array(final_result['z'])
        #     final_result['visibility'] = np.array(final_result['visibility'])

    
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
final_results_dict = {'head':final_results_nose,'nose':final_results_nose,
                            'l_shoulder':final_results_l_shoulder,'l_elbow':final_results_l_elbow,'l_wrist':final_results_l_wrist,
                            'r_shoulder':final_results_r_shoulder,'r_elbow':final_results_r_elbow,'r_wrist':final_results_r_wrist,
                            'l_hip':final_results_l_hip,'l_knee':final_results_l_knee,'l_ankle':final_results_l_ankle,
                            'r_hip':final_results_r_hip,'r_knee':final_results_r_knee,'r_ankle':final_results_r_ankle}
                            

# print(final_results_dict['nose'])              

print('CALCULATING ANGLES...',f'total frames = {frame_ct}')


# a_results,b_results,c_results,reba_class = calc_and_plot_reba_w_lib(final_results_dict,plot=True)


    
# Calculate BARD

# shoulder_vec = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_r_shoulder['x']),
# np.array(final_results_l_shoulder['y'])-np.array(final_results_r_shoulder['y']),
# np.array(final_results_l_shoulder['z'])-np.array(final_results_r_shoulder['z'])])

# camera_vec = 

#angle calculation

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

# sagital_plane_l_shoulder_angle,frontal_plane_l_should_angle = get_sagital_and_frontal_angles()
test_frame = 181
print('l_shoulder_sagital_angle',l_upper_arm_sagital_angles[test_frame])
print('l_shoulder_frontal_angle',l_upper_arm_frontal_angles[test_frame])
print('r_shoulder_sagital_angle',r_upper_arm_sagital_angles[test_frame])
print('r_shoulder_frontal_angle',r_upper_arm_frontal_angles[test_frame])


plot_ref_points_w_body(final_results_dict,l_projected_ref_points+r_projected_ref_points,test_frame)

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

# v1_l_shoulder = np.array([np.array(l_shoulder_hip_ref_points[:,0])-np.array(final_results_l_shoulder['x']),
# np.array(l_shoulder_hip_ref_points[:,1])-np.array(final_results_l_shoulder['y']),
# np.array(l_shoulder_hip_ref_points[:,2])-np.array(final_results_l_shoulder['z'])])

# v2_l_shoulder =  np.array([np.array(final_results_l_elbow['x'])-np.array(final_results_l_shoulder['x']),
# np.array(final_results_l_elbow['y'])-np.array(final_results_l_shoulder['y']),
# np.array(final_results_l_elbow['z'])-np.array(final_results_l_shoulder['z'])])

# v1_r_shoulder = np.array([np.array(r_shoulder_hip_ref_points[:,0])-np.array(final_results_r_shoulder['x']),
# np.array(r_shoulder_hip_ref_points[:,0])-np.array(final_results_r_shoulder['y']),
# np.array(r_shoulder_hip_ref_points[:,0])-np.array(final_results_r_shoulder['z'])])

# v2_r_shoulder =  np.array([np.array(final_results_l_elbow['x'])-np.array(final_results_r_shoulder['x']),
# np.array(final_results_l_elbow['y'])-np.array(final_results_r_shoulder['y']),
# np.array(final_results_l_elbow['z'])-np.array(final_results_r_shoulder['z'])])

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

np.ones_like
angle_deg_neck = get_joint_angles(v1_neck,v2_neck)
angle_deg_neck = 180-angle_deg_neck
angle_deg_neck = angle_deg_neck-(np.ones_like(angle_deg_neck)*angle_deg_neck[0])
# angle_deg_l_shoulder = get_joint_angles(v1_l_shoulder,v2_l_shoulder)
# angle_deg_r_shoulder = get_joint_angles(v1_r_shoulder,v2_r_shoulder)

# angle_deg_l_hip,angle_rad_l_hip = get_joint_angles(v1_l_hip,v2_l_hip)
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



a_results,b_results,c_results,reba_class,upper_arm_scores,lower_arm_scores,wrist_scores,neck_scores,trunk_scores,leg_scores =calc_reba_custom(reba_angles,plot=False)
reba_angles_dataframe['a_score'] = a_results
reba_angles_dataframe['b_score'] = b_results
reba_angles_dataframe['c_score'] = c_results

reba_angles_dataframe['upper_arm_score'] = upper_arm_scores
reba_angles_dataframe['lower_arm_score'] = lower_arm_scores
reba_angles_dataframe['wrist_score'] = wrist_scores
reba_angles_dataframe['neck_score'] = neck_scores
reba_angles_dataframe['trunk_score'] = trunk_scores
reba_angles_dataframe['leg_score'] = leg_scores


dataframe = pd.DataFrame(data=reba_angles_dataframe)

dataframe.to_csv('reba_test_videos/data/reba_data.csv')

#These calcualte the distance between nodes
# nose_ankle_dist = calc_distance_between_nodes(final_results_nose,final_results_r_ankle)
# nose_mid_hip_dist = calc_distance_between_nodes(final_results_nose,mid_hip)
# nose_l_knee_dist = calc_distance_between_nodes(final_results_nose,final_results_l_knee)
# nose_l_shoulder_dist = calc_distance_between_nodes(final_results_nose,final_results_l_shoulder)


# print(angle_deg_hip,angle_rad_hip)



# plt.figure(1)
# plt.title('hip angle tracking')
# plt.plot(range(len(angle_rad)),angle_rad,color='r', label='angle rad')
# plt.xlabel("Frame")
# plt.ylabel("angle (rad)")
# plt.legend()

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


