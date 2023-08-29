import random
from utils import *
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt
from ergonomics.reba import RebaScore
from reba_score_class import RebaScoreMIT

test = True
front_view = True

if test:
    input_video_name = 'booker_trimmed.mp4'
    # input_video_name = 'booker_test.MOV'
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
print(sys.argv)
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

final_results_head = {'x':[],'y':[],'z':[],'visibility':[]}

frame_ct = 0
second_processed_count = 0
print_update = True
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    # if frame_ct == 0: # this plots the pose estimation in the 3d coordinate system
    #     print('Nose world landmark:'),
    #     print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    #     mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    frame_ct += 1
    if frame_ct%camera_frames_per_second == 0:
        second_processed_count += 1
        print_update = True
    if second_processed_count %give_processing_updates_time == 0 and print_update :
        print(f'{second_processed_count} seconds of video processed...') 
        print_update = False
    
    try: 
        nose = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
 
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

hip_angle_ref_point = np.array([0,-5,0])

v1_l_hip = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_l_hip['x']),
np.array(final_results_l_shoulder['y'])-np.array(final_results_l_hip['y']),
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

v1_l_shoulder = np.array([np.array(final_results_l_hip['x'])-np.array(final_results_l_shoulder['x']),
np.array(final_results_l_hip['y'])-np.array(final_results_l_shoulder['y']),
np.array(final_results_l_hip['z'])-np.array(final_results_l_shoulder['z'])])

v2_l_shoulder =  np.array([np.array(final_results_l_elbow['x'])-np.array(final_results_l_shoulder['x']),
np.array(final_results_l_elbow['y'])-np.array(final_results_l_shoulder['y']),
np.array(final_results_l_elbow['z'])-np.array(final_results_l_shoulder['z'])])

v1_r_shoulder = np.array([np.array(final_results_r_hip['x'])-np.array(final_results_r_shoulder['x']),
np.array(final_results_r_hip['y'])-np.array(final_results_r_shoulder['y']),
np.array(final_results_r_hip['z'])-np.array(final_results_r_shoulder['z'])])

v2_r_shoulder =  np.array([np.array(final_results_l_elbow['x'])-np.array(final_results_r_shoulder['x']),
np.array(final_results_l_elbow['y'])-np.array(final_results_r_shoulder['y']),
np.array(final_results_l_elbow['z'])-np.array(final_results_r_shoulder['z'])])

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

v2_l_wrist =  np.array([np.array(final_results_l_index['x'])-np.array(final_results_l_wrist['x']),
np.array(final_results_l_index['y'])-np.array(final_results_l_wrist['y']),
np.array(final_results_l_index['z'])-np.array(final_results_l_wrist['z'])])

v1_r_wrist = np.array([np.array(final_results_r_elbow['x'])-np.array(final_results_r_wrist['x']),
np.array(final_results_r_elbow['y'])-np.array(final_results_r_wrist['y']),
np.array(final_results_r_elbow['z'])-np.array(final_results_r_wrist['z'])])

v2_r_wrist =  np.array([np.array(final_results_r_index['x'])-np.array(final_results_r_wrist['x']),
np.array(final_results_r_index['y'])-np.array(final_results_r_wrist['y']),
np.array(final_results_r_index['z'])-np.array(final_results_r_wrist['z'])])

# mid_shoulders = calc_midpoint_vec(final_results_l_shoulder,final_results_r_shoulder)
# mid_hips = calc_midpoint_vec(final_results_l_hip,final_results_r_hip)

angle_deg_neck,angle_rad_neck = get_joint_angles(v1_neck,v2_neck)
angle_deg_neck = 180-angle_deg_neck
angle_deg_l_shoulder,angle_rad_l_shoulder = get_joint_angles(v1_l_shoulder,v2_l_shoulder)
angle_deg_r_shoulder,angle_rad_r_shoulder = get_joint_angles(v1_r_shoulder,v2_r_shoulder)

angle_deg_l_hip,angle_rad_l_hip = get_joint_angles(v1_l_hip,v2_l_hip)
angle_deg_mid_hip,angle_rad_mid_hip = get_joint_angles(v1_mid_hip,v2_mid_hip)
mid_hip_visibility_score_avg = (np.array(final_results_l_hip['visibility']) + np.array(final_results_r_hip['visibility'])  + np.array(final_results_l_shoulder['visibility']) + np.array(final_results_r_shoulder['visibility']) )/4
angle_deg_l_knee,angle_rad_l_knee = get_joint_angles(v1_l_knee,v2_l_knee)
angle_deg_r_knee,angle_rad_r_knee = get_joint_angles(v1_r_knee,v2_r_knee)
angle_deg_l_knee = 180-angle_deg_l_knee
angle_deg_r_knee = 180-angle_deg_r_knee

angle_deg_l_elbow,angle_rad_l_elbow = get_joint_angles(v1_l_elbow,v2_l_elbow)
angle_deg_r_elbow,angle_rad_r_elbow = get_joint_angles(v1_r_elbow,v2_r_elbow)
angle_deg_r_elbow = 180-angle_deg_r_elbow
angle_deg_l_elbow = 180-angle_deg_l_elbow

angle_deg_l_wrist,angle_rad_l_wrist = get_joint_angles(v1_l_wrist,v2_l_wrist)
angle_deg_r_wrist,angle_rad_r_wrist = get_joint_angles(v1_r_wrist,v2_r_wrist)
angle_deg_l_wrist = 180-angle_deg_l_wrist-10
angle_deg_r_wrist = 180-angle_deg_r_wrist-10

reba_angles = {'l_upper_arm':angle_deg_l_shoulder,'r_upper_arm':angle_deg_r_shoulder,'l_lower_arm':angle_deg_l_elbow,'r_lower_arm':angle_deg_r_elbow,
                    'l_wrist':angle_deg_l_wrist,'r_wrist':angle_deg_r_wrist,
                    'l_knee':angle_deg_l_knee,'r_knee':angle_deg_r_knee,'neck':angle_deg_neck,'trunk_angle':angle_deg_mid_hip}

calc_reba_custom(reba_angles,plot=True)

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

# print(mid_hip_visibility_score_avg)
plt.figure(1)
plt.subplot(2,6,1)
plt.plot(range(len(angle_deg_mid_hip)),angle_deg_mid_hip,color='r', label='angle deg')
plt.title('REBA Trunk angles')

plt.subplot(2,6,2)
plt.plot(range(len(mid_hip_visibility_score_avg)),mid_hip_visibility_score_avg)
plt.title('REBA Trunk angles visibility')

plt.subplot(2,6,3)
plt.plot(range(len(angle_deg_l_knee)),angle_deg_l_knee)
plt.title('REBA L Knee angles')
plt.subplot(2,6,4)
plt.plot(range(len(angle_deg_r_knee)),angle_deg_r_knee)
plt.title('REBA R Knee angles')

plt.subplot(2,6,5)
plt.plot(range(len(angle_deg_l_shoulder)),angle_deg_l_shoulder)
plt.title('REBA L Shoulder angles')
plt.subplot(2,6,6)
plt.plot(range(len(angle_deg_l_shoulder)),angle_deg_l_shoulder)
plt.title('REBA R Shoulder angles')

plt.subplot(2,6,7)
plt.plot(range(len(angle_deg_l_elbow)),angle_deg_l_elbow)
plt.title('REBA L elbow angles')
plt.subplot(2,6,8)
plt.plot(range(len(angle_deg_r_elbow)),angle_deg_r_elbow)
plt.title('REBA R elbow angles')

plt.subplot(2,6,9)
plt.plot(range(len(angle_deg_neck)),angle_deg_neck)
plt.title('REBA neck angles')

plt.subplot(2,6,10)
plt.plot(range(len(angle_deg_l_wrist)),angle_deg_l_wrist)
plt.title('REBA l wrist angles')

plt.subplot(2,6,11)
plt.plot(range(len(angle_deg_r_wrist)),angle_deg_r_wrist)
plt.title('REBA r wrist angles')

plt.show()
# axis[1].xlabel("Frame")
# axis[1].ylabel("angle (deg)")
# plt.plot(range(len(angle_deg_l_hip)),angle_deg_l_hip,color='r', label='angle deg')
# plt.subplot(range(len(angle_deg_mid_hip)),angle_deg_mid_hip,color='m', label='angle deg')
# plt.subplot(range(len(mid_hip_visibility_score_avg)),mid_hip_visibility_score_avg)
# plt.plot(range(len(angle_deg_elbow)),angle_deg_elbow,color='b', label='angle deg')
# plt.plot(range(len(angle_deg_knee)),angle_deg_knee,color='g', label='angle deg')
# plt.plot(range(len(angle_deg_neck)),angle_deg_neck,color='k', label='angle deg')
# plt.xlabel("Frame")
# plt.ylabel("angle (deg)")
# plt.legend(['l hip','mid hip','elbow','knee','neck'])

# plt.figure(2)
# plt.title('point distances')
# plt.plot(range(len(nose_ankle_dist)),nose_ankle_dist,color='r', label='angle deg')
# plt.plot(range(len(nose_l_knee_dist)),nose_l_knee_dist,color='b', label='angle deg')
# plt.plot(range(len(nose_mid_hip_dist)),nose_mid_hip_dist,color='g', label='angle deg')
# plt.plot(range(len(nose_l_shoulder_dist)),nose_l_shoulder_dist,color='k', label='angle deg')
# plt.xlabel("Frame")
# plt.ylabel("distance (m)")
# plt.legend(['nose -> ankle','nose -> l knee','nose -> mid hip','nose -> l shoulder'])

# plt.figure(3)
# plt.title('raw x values plot')
# plt.plot(range(len(final_results_l_ankle['x'])),final_results_l_ankle['x'],color='r', label='angle deg')
# plt.xlabel("Frame")
# plt.ylabel("position (m)")
# plt.legend(['left ankle x'])

# plt.figure(4)
# plt.title('Visibility Score for Markers (upper body)')
# legend = []
# for i,key  in enumerate(final_results_dict):
#     if i<6:
#         plt.plot(range(len(final_results_dict[key]['visibility'])),final_results_dict[key]['visibility'])
#         legend.append(key)
# plt.legend(legend)

# plt.figure(5)
# plt.title('Visibility Score for Markers (lower body)')
# legend = []
# for i,key  in enumerate(final_results_dict):
#     if i>6:
#         plt.plot(range(len(final_results_dict[key]['visibility'])),final_results_dict[key]['visibility'])
#         legend.append(key)
# plt.legend(legend)

plt.show()
