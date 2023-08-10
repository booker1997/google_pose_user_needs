import random
from utils import *
import cv2
import mediapipe as mp
import numpy as np
import sys
from matplotlib import pyplot as plt

test = False
if test:
    input_video_name = 'booker_trimmed.mp4'
    give_processing_updates_time = 1
else:
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


final_results_l_shoulder = {'x':[],'y':[],'z':[]}
final_results_l_hip = {'x':[],'y':[],'z':[]}
final_results_l_knee = {'x':[],'y':[],'z':[]}
final_results_r_knee = {'x':[],'y':[],'z':[]}
final_results_r_shoulder = {'x':[],'y':[],'z':[]}
final_results_r_hip = {'x':[],'y':[],'z':[]}
final_results_nose = {'x':[],'y':[],'z':[]}
final_results_l_wrist = {'x':[],'y':[],'z':[]}
final_results_r_wrist = {'x':[],'y':[],'z':[]}
final_results_l_elbow = {'x':[],'y':[],'z':[]}
final_results_r_elbow = {'x':[],'y':[],'z':[]}
final_results_l_ankle = {'x':[],'y':[],'z':[]}
final_results_r_ankle = {'x':[],'y':[],'z':[]}

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
        
        final_results_l_shoulder['x'].append(l_shoulder.x)
        final_results_l_shoulder['y'].append(l_shoulder.y)
        final_results_l_shoulder['z'].append(l_shoulder.z)

        final_results_r_shoulder['x'].append(r_shoulder.x)
        final_results_r_shoulder['y'].append(r_shoulder.y)
        final_results_r_shoulder['z'].append(r_shoulder.z)

        final_results_l_hip['x'].append(l_hip.x)
        final_results_l_hip['y'].append(l_hip.y)
        final_results_l_hip['z'].append(l_hip.z)

        final_results_r_hip['x'].append(r_hip.x)
        final_results_r_hip['y'].append(r_hip.y)
        final_results_r_hip['z'].append(r_hip.z)

        final_results_l_knee['x'].append(l_knee.x)
        final_results_l_knee['y'].append(l_knee.y)
        final_results_l_knee['z'].append(l_knee.z)

        final_results_r_knee['x'].append(r_knee.x)
        final_results_r_knee['y'].append(r_knee.y)
        final_results_r_knee['z'].append(r_knee.z)

        final_results_l_ankle['x'].append(l_ankle.x)
        final_results_l_ankle['y'].append(l_ankle.y)
        final_results_l_ankle['z'].append(l_ankle.z)

        final_results_r_ankle['x'].append(r_ankle.x)
        final_results_r_ankle['y'].append(r_ankle.y)
        final_results_r_ankle['z'].append(r_ankle.z)

        final_results_nose['x'].append(nose.x)
        final_results_nose['y'].append(nose.y)
        final_results_nose['z'].append(nose.z)

        final_results_l_elbow['x'].append(l_elbow.x)
        final_results_l_elbow['y'].append(l_elbow.y)
        final_results_l_elbow['z'].append(l_elbow.z)

        final_results_r_elbow['x'].append(r_elbow.x)
        final_results_r_elbow['y'].append(r_elbow.y)
        final_results_r_elbow['z'].append(r_elbow.z)

        final_results_l_wrist['x'].append(l_wrist.x)
        final_results_l_wrist['y'].append(l_wrist.y)
        final_results_l_wrist['z'].append(l_wrist.z)

        final_results_r_wrist['x'].append(r_wrist.x)
        final_results_r_wrist['y'].append(r_wrist.y)
        final_results_r_wrist['z'].append(r_wrist.z)

        

    
    # Plot pose world landmarks.
        # if random.random() < 0.05:
        #     plot = mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    except:
        pass

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(image)
pose.close()
cap.release()
out.release()

print('CALCULATING ANGLES...',f'total frames = {frame_ct}')
#angle calculation

mid_hip = calc_midpoint_vec(final_results_l_hip,final_results_r_hip)
mid_shoulder = calc_midpoint_vec(final_results_l_shoulder,final_results_r_shoulder)
mid_knee = calc_midpoint_vec(final_results_l_knee,final_results_r_knee)

v1_l_hip = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_l_hip['x']),
np.array(final_results_l_shoulder['y'])-np.array(final_results_l_hip['y']),
np.array(final_results_l_shoulder['z'])-np.array(final_results_l_hip['z'])])

v2_l_hip =  np.array([np.array(final_results_l_knee['x'])-np.array(final_results_l_hip['x']),
np.array(final_results_l_knee['y'])-np.array(final_results_l_hip['y']),
np.array(final_results_l_knee['z'])-np.array(final_results_l_hip['z'])])

v1_mid_hip = np.array([np.array(mid_shoulder['x'])-np.array(mid_hip['x']),
np.array(mid_shoulder['y'])-np.array(mid_hip['y']),
np.array(mid_shoulder['z'])-np.array(mid_hip['z'])])

v2_mid_hip =  np.array([np.array(mid_knee['x'])-np.array(mid_hip['x']),
np.array(mid_knee['y'])-np.array(mid_hip['y']),
np.array(mid_knee['z'])-np.array(mid_hip['z'])])

v1_l_elbow = np.array([np.array(final_results_l_shoulder['x'])-np.array(final_results_l_elbow['x']),
np.array(final_results_l_shoulder['y'])-np.array(final_results_l_elbow['y']),
np.array(final_results_l_shoulder['z'])-np.array(final_results_l_elbow['z'])])

v2_l_elbow =  np.array([np.array(final_results_l_wrist['x'])-np.array(final_results_l_elbow['x']),
np.array(final_results_l_wrist['y'])-np.array(final_results_l_elbow['y']),
np.array(final_results_l_wrist['z'])-np.array(final_results_l_elbow['z'])])

v1_l_knee = np.array([np.array(final_results_l_hip['x'])-np.array(final_results_l_knee['x']),
np.array(final_results_l_hip['y'])-np.array(final_results_l_knee['y']),
np.array(final_results_l_hip['z'])-np.array(final_results_l_knee['z'])])

v2_l_knee =  np.array([np.array(final_results_l_ankle['x'])-np.array(final_results_l_knee['x']),
np.array(final_results_l_ankle['y'])-np.array(final_results_l_knee['y']),
np.array(final_results_l_ankle['z'])-np.array(final_results_l_knee['z'])])

v1_neck = np.array([np.array(final_results_nose['x'])-np.array(mid_shoulder['x']),
np.array(final_results_nose['y'])-np.array(mid_shoulder['y']),
np.array(final_results_nose['z'])-np.array(mid_shoulder['z'])])

v2_neck =  np.array([np.array(mid_hip['x'])-np.array(mid_shoulder['x']),
np.array(mid_hip['y'])-np.array(mid_shoulder['y']),
np.array(mid_hip['z'])-np.array(mid_shoulder['z'])])

# mid_shoulders = calc_midpoint_vec(final_results_l_shoulder,final_results_r_shoulder)
# mid_hips = calc_midpoint_vec(final_results_l_hip,final_results_r_hip)

angle_deg_neck,angle_rad_neck = get_joint_angles(v1_neck,v2_neck)
angle_deg_l_hip,angle_rad_l_hip = get_joint_angles(v1_l_hip,v2_l_hip)
angle_deg_mid_hip,angle_rad_mid_hip = get_joint_angles(v1_mid_hip,v2_mid_hip)
angle_deg_knee,angle_rad_knee = get_joint_angles(v1_l_knee,v2_l_knee)
angle_deg_elbow,angle_rad_elbow = get_joint_angles(v1_l_elbow,v2_l_elbow)

nose_ankle_dist = calc_distance_between_nodes(final_results_nose,final_results_r_ankle)
nose_mid_hip_dist = calc_distance_between_nodes(final_results_nose,mid_hip)
nose_l_knee_dist = calc_distance_between_nodes(final_results_nose,final_results_l_knee)
nose_l_shoulder_dist = calc_distance_between_nodes(final_results_nose,final_results_l_shoulder)


# print(angle_deg_hip,angle_rad_hip)



# plt.figure(1)
# plt.title('hip angle tracking')
# plt.plot(range(len(angle_rad)),angle_rad,color='r', label='angle rad')
# plt.xlabel("Frame")
# plt.ylabel("angle (rad)")
# plt.legend()

plt.figure(1)
plt.title('hip angle tracking deg')
plt.plot(range(len(angle_deg_l_hip)),angle_deg_l_hip,color='r', label='angle deg')
plt.plot(range(len(angle_deg_mid_hip)),angle_deg_mid_hip,color='m', label='angle deg')
plt.plot(range(len(angle_deg_elbow)),angle_deg_elbow,color='b', label='angle deg')
plt.plot(range(len(angle_deg_knee)),angle_deg_knee,color='g', label='angle deg')
plt.plot(range(len(angle_deg_neck)),angle_deg_neck,color='k', label='angle deg')
plt.xlabel("Frame")
plt.ylabel("angle (deg)")
plt.legend(['l hip','mid hip','elbow','knee','neck'])

plt.figure(2)
plt.title('point distances')
plt.plot(range(len(nose_ankle_dist)),nose_ankle_dist,color='r', label='angle deg')
plt.plot(range(len(nose_l_knee_dist)),nose_l_knee_dist,color='b', label='angle deg')
plt.plot(range(len(nose_mid_hip_dist)),nose_mid_hip_dist,color='g', label='angle deg')
plt.plot(range(len(nose_l_shoulder_dist)),nose_l_shoulder_dist,color='k', label='angle deg')
plt.xlabel("Frame")
plt.ylabel("distance (m)")
plt.legend(['nose -> ankle','nose -> l knee','nose -> mid hip','nose -> l shoulder'])

plt.figure(3)
plt.title('raw x values plot')
plt.plot(range(len(final_results_l_ankle['x'])),final_results_l_ankle['x'],color='r', label='angle deg')
plt.xlabel("Frame")
plt.ylabel("position (m)")
plt.legend(['left ankle x'])

# plt.title('L elbow angle tracking deg')
# plt.plot(range(len(angle_deg_elbow)),angle_deg_elbow,color='r', label='angle deg')
# plt.xlabel("Frame")
# plt.ylabel("angle (deg)")
# plt.legend()


# plt.title('L elbow angle tracking deg')
# plt.plot(range(len(angle_deg_elbow)),angle_deg_elbow,color='r', label='angle deg')
# plt.xlabel("Frame")
# plt.ylabel("angle (deg)")
# plt.legend()
# plt.figure(1)
# plt.title('Left Shoulder Tracking')
# plt.plot(range(len(final_results_l_shoulder['x'][:110])),final_results_l_shoulder['x'][:110],color='r', label='x')
# plt.plot(range(len(final_results_l_shoulder['y'][:110])),final_results_l_shoulder['y'][:110],color='g', label='y')
# plt.plot(range(len(final_results_l_shoulder['z'][:110])),final_results_l_shoulder['z'][:110],color='b', label='z')
# plt.xlabel("Frame")
# plt.ylabel("Coordinate (m)")
# plt.legend()

# plt.figure(2)
# plt.title('Left hip Tracking')
# plt.plot(range(len(final_results_l_hip['x'][:110])),final_results_l_hip['x'][:110],color='r', label='x')
# plt.plot(range(len(final_results_l_hip['y'][:110])),final_results_l_hip['y'][:110],color='g', label='y')
# plt.plot(range(len(final_results_l_hip['z'][:110])),final_results_l_hip['z'][:110],color='b', label='z')
# plt.xlabel("Frame")
# plt.ylabel("Coordinate (m)")
# plt.legend()

# plt.figure(3)
# plt.title('Left knee Tracking')
# plt.plot(range(len(final_results_l_knee['x'][:110])),final_results_l_knee['x'][:110],color='r', label='x')
# plt.plot(range(len(final_results_l_knee['y'][:110])),final_results_l_knee['y'][:110],color='g', label='y')
# plt.plot(range(len(final_results_l_knee['z'][:110])),final_results_l_knee['z'][:110],color='b', label='z')
# plt.xlabel("Frame")
# plt.ylabel("Coordinate (m)")
# plt.legend()
plt.show()
