import csv
# import mediapipe as mp
import numpy as np
# import sys
from matplotlib import pyplot as plt
from ergonomics.reba import RebaScore
from reba_score_class import RebaScoreMIT
import plotly.express as px
import plotly.graph_objects as go
import ast



def remake_dicts_from_csv(filename):
    """Loads a CSV file as a dictionary by columns.

  Args:
    filename: The path to the CSV file.

  Returns:
    A dictionary where the keys are the column names and the values are lists
    of the values in that column.
  """

    with open(filename, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = {}
        for row in reader:
            for i, header in enumerate(headers):
                if header == 'peak_risk_level':
                    data.setdefault(header, []).append(row[i])
                else:
                    data.setdefault(header, []).append(ast.literal_eval(row[i]))
        return data

def get_sagital_and_frontal_angles(plane_start_joint,on_plane_angle_start_joint,on_plane_angle_mid_joint,angle_end_joint,hip_ref_points = []):
    if len(hip_ref_points)>0:
        frontal_angles = []
        sagital_angles = []
        proj_point_sagital = []
        proj_point_frontal = []
        for frame in range(len(plane_start_joint['x'])):
            angle_start_and_mid_joint_frontal_plane_normal_vec,angle_start_and_mid_joint_l_shoulder_hip_frontal_plane_D,angle_mid_joint_sagital_plane_normal_vec,angle_mid_joint_sagital_plane_D= define_frontal_and_sagital_planes(np.array([plane_start_joint['x'][frame],plane_start_joint['y'][frame],plane_start_joint['z'][frame]]),
                                                                                                            np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                                            np.array([on_plane_angle_start_joint['x'][frame],on_plane_angle_start_joint['y'][frame],on_plane_angle_start_joint['z'][frame]]),
                                                                                                            np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))
                                                                                                                                                                                            
            angle_end_joint_projected_on_frontal_plane = project_point_on_plane(angle_start_and_mid_joint_frontal_plane_normal_vec,angle_start_and_mid_joint_l_shoulder_hip_frontal_plane_D,np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))
            angle_end_joint_projected_on_sagital_plane = project_point_on_plane(angle_mid_joint_sagital_plane_normal_vec,angle_mid_joint_sagital_plane_D,np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))

            frontal_angle = get_single_joint_angles(angle_end_joint_projected_on_frontal_plane-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                                    hip_ref_points[frame]-np.array([on_plane_angle_mid_joint['x'][frame],
                                                                                                        on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]))
            sagital_angle = get_single_joint_angles(angle_end_joint_projected_on_sagital_plane-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                    hip_ref_points[frame]-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]))

            frontal_angles.append(frontal_angle)
            sagital_angles.append(sagital_angle)
            proj_point_sagital.append(angle_end_joint_projected_on_sagital_plane)
            proj_point_frontal.append(angle_end_joint_projected_on_frontal_plane)
        return frontal_angles,sagital_angles,[proj_point_sagital,proj_point_frontal]
    else:
        frontal_angles = []
        sagital_angles = []
        proj_point_sagital = []
        proj_point_frontal = []
        for frame in range(len(plane_start_joint['x'])):
            angle_start_and_mid_joint_frontal_plane_normal_vec,angle_start_and_mid_joint_l_shoulder_hip_frontal_plane_D,angle_mid_joint_sagital_plane_normal_vec,angle_mid_joint_sagital_plane_D= define_frontal_and_sagital_planes(np.array([plane_start_joint['x'][frame],plane_start_joint['y'][frame],plane_start_joint['z'][frame]]),
                                                                                                            np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                                            np.array([on_plane_angle_start_joint['x'][frame],on_plane_angle_start_joint['y'][frame],on_plane_angle_start_joint['z'][frame]]),
                                                                                                            np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))
                                                                                                                                                                                            
            angle_end_joint_projected_on_frontal_plane = project_point_on_plane(angle_start_and_mid_joint_frontal_plane_normal_vec,angle_start_and_mid_joint_l_shoulder_hip_frontal_plane_D,np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))
            angle_end_joint_projected_on_sagital_plane = project_point_on_plane(angle_mid_joint_sagital_plane_normal_vec,angle_mid_joint_sagital_plane_D,np.array([angle_end_joint['x'][frame],angle_end_joint['y'][frame],angle_end_joint['z'][frame]]))

            frontal_angle = get_single_joint_angles(angle_end_joint_projected_on_frontal_plane-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                                    np.array([on_plane_angle_start_joint['x'][frame],on_plane_angle_start_joint['y'][frame],on_plane_angle_start_joint['z'][frame]])-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]))
            sagital_angle = get_single_joint_angles(angle_end_joint_projected_on_sagital_plane-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]),
                                                                                    np.array([on_plane_angle_start_joint['x'][frame],on_plane_angle_start_joint['y'][frame],on_plane_angle_start_joint['z'][frame]])-np.array([on_plane_angle_mid_joint['x'][frame],on_plane_angle_mid_joint['y'][frame],on_plane_angle_mid_joint['z'][frame]]))

            frontal_angles.append(frontal_angle)
            sagital_angles.append(sagital_angle)
            proj_point_sagital.append(angle_end_joint_projected_on_sagital_plane)
            proj_point_frontal.append(angle_end_joint_projected_on_frontal_plane)
        return frontal_angles,sagital_angles,[proj_point_sagital,proj_point_frontal]

def define_frontal_and_sagital_planes(point1,point2,point3,point_to_project,plot_planes = False):
    # point1 = np.array([x1, y1, z1])
    # point2 = np.array([x2, y2, z2])
    # point3 = np.array([x3, y3, z3])
    pts = np.array([point1,point2,point3])

    # Calculate two vectors on the plane
    vector1 = point1 - point2 
    vector2 = point3 -point2 

    #calculate eq of plane with normal
    normal_vector = np.cross(vector1, vector2)
    normal_vector /= np.linalg.norm(normal_vector)
    A, B, C = normal_vector
    D = np.dot(normal_vector, point2)
    # print('ABCD',A,B,C,D)

   
    #project point onto plane
    # point_to_project = np.array([.5,.5,.5]) #test for projection
    projected_point = project_point_on_plane(plane_norm_vec=normal_vector,
                                            plane_d=D,point_to_project=point_to_project)
                                            # vecs_on_plane=(vector1,vector2),
                                            # ref_point_on_plane=point2,

    
    # Now, you have the equation of the first plane: Ax + By + Cz + D = 0

    # To find a plane perpendicular to the first plane and passing through two points from the first plane
    # Let's say you want to use point1 and point2 from the first plane

    # Calculate the normal vector of the new plane by taking the cross product of the normal_vector and vector1
    new_normal_vector = np.cross(normal_vector, vector2)

    # Normalize the new normal vector to get a unit normal vector
    new_normal_vector /= np.linalg.norm(new_normal_vector)
    

    new_A,new_B,new_C = new_normal_vector
    # Calculate the D coefficient for the new plane using one of the points (e.g., point1)
    new_D = np.dot(new_normal_vector, point2)
    # print('vec2',vector2)
    # print('ABCDnew',new_A,new_B,new_C,new_D)
    # print('dot product of norms', np.dot(new_normal_vector,normal_vector))

    proj_point_on_new_plane = project_point_on_plane(new_normal_vector,new_D,point_to_project)

    if plot_planes:
        # Define the range of x and y values
        x = np.linspace(-.5, .5, 100)
        y = np.linspace(-.5, .5, 100)
        origin = np.zeros((3,1))
        # Generate a grid of (x, y) coordinates
        X, Y = np.meshgrid(x, y)
        # Calculate z values
        Z = (D - A*X - B*Y) / C
        Z_new = (new_D - new_A*X - new_B*Y) / new_C

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, color='b', alpha=0.5)
        ax.plot_surface(X, Y, Z_new, color='r', alpha=0.5)
        markers = ['.','*','^']
        for i,point in enumerate(pts):
            ax.scatter(point[0], point[1], point[2],marker=markers[i])
        ax.scatter(point_to_project[0], point_to_project[1], point_to_project[2],marker=markers[i],c='b') 
        ax.scatter(projected_point[0], projected_point[1], projected_point[2],marker=markers[i],c='g')  
        ax.scatter(proj_point_on_new_plane[0], proj_point_on_new_plane[1], proj_point_on_new_plane[2],marker=markers[i],c='r')  
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        # Show the plot
        plt.show()

    # Now, you have the equation of the second plane perpendicular to the first plane and passing through point1 and point2.
    # Its equation is: new_Ax + new_By + new_Cz + new_D = 0

    return normal_vector,D,new_normal_vector,new_D



def project_point_on_plane(plane_norm_vec,plane_d,point_to_project,vecs_on_plane=(None,None),ref_point_on_plane=None):
    if vecs_on_plane[0]!=None and vecs_on_plane[1] !=None:
        #calculate plane 1 transformation matrix with vecs
        plane_1_v_matrix = np.transpose(np.array([vecs_on_plane[0],vecs_on_plane[1]]))
        plane_1_sudo_inv = np.linalg.pinv(plane_1_v_matrix)
        plane_1_transformation_matrix = np.matmul(plane_1_v_matrix,plane_1_sudo_inv)
        
        #project point onto plane
        projected_vector = np.matmul(plane_1_transformation_matrix,point_to_project- ref_point_on_plane)
        #print('PROJ DOT',np.dot(projected_vector,point_to_project- ref_point_on_plane)) # this should be 0
        projected_point = ref_point_on_plane + projected_vector
    else:
        A,B,C = plane_norm_vec
        D = plane_d
        x_vals = np.random.rand(3)
        y_vals = np.random.rand(3)
        z_vals = (D - A*x_vals - B*y_vals) / C
        point1 = np.array([x_vals[0],y_vals[0],z_vals[0]])
        point2 = np.array([x_vals[1],y_vals[1],z_vals[1]])
        point3 = np.array([x_vals[2],y_vals[2],z_vals[2]])

        vector1 = point1 - point2 
        vector2 = point3 -point2

        plane_1_v_matrix = np.transpose(np.array([vector1,vector2]))
        plane_1_sudo_inv = np.linalg.pinv(plane_1_v_matrix)
        plane_1_transformation_matrix = np.matmul(plane_1_v_matrix,plane_1_sudo_inv)
        
        #project point onto plane
        projected_vector = np.matmul(plane_1_transformation_matrix,point_to_project- point1)
        #print('PROJ DOT',np.dot(projected_vector,point_to_project- ref_point_on_plane)) # this should be 0
        projected_point = point1 + projected_vector

    return projected_point
    

def get_single_joint_angles(v1,v2):
    v1mag = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2]  * v1[2] )
    v1norm = [v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag]

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2]  * v2[2] )
    v2norm = [v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag]

    res = (v1norm[0] * v2norm[0]) + (v1norm[1] * v2norm[1]) + (v1norm[2] * v2norm[2])

    angle_rad = np.arccos(res)
    full_angle_deg = np.rad2deg(angle_rad)
    # print('HERE',full_angle_deg)

    # for xy plane
    # xy_vec_a = np.array([v1[0],v1[1]])
    # xy_vec_b = np.array([v2[0],v2[1]])
    
    # xz_vec_a = np.array([v1[0],v1[2]])
    # xz_vec_b = np.array([v2[0],v2[2]])

    # yz_vec_a = np.array([v1[1],v1[2]])
    # yz_vec_b = np.array([v2[1],v2[2]])



    return full_angle_deg

def find_point_on_line(final_results_joint_1,final_results_joint_2,final_results_joint_to_find_on_line):
    
    def find_point(x_1,y_1,z_1,x_2,y_2,z_2,x_p,y_p,z_p):#,delta_x,delta_y,delta_z):
        # returns t

        # Define points A, B, and C
        point_A = np.array([x_1, y_1, z_1])
        point_B = np.array([x_2, y_2, z_2])
        point_C = np.array([x_p, y_p, z_p])

        # Calculate the direction vector of the line AB
        direction_vector_AB = point_B - point_A

        # Calculate the vector from A to C
        vector_AC = point_C - point_A

        # Calculate the projection of vector AC onto vector AB
        projection_AC_AB = np.dot(vector_AC, direction_vector_AB) / np.dot(direction_vector_AB, direction_vector_AB)

        # Calculate the point D on line AB such that CD is perpendicular to AB
        point_D = point_A + projection_AC_AB * direction_vector_AB

        # print("Point D:", point_D)
        
        return point_D

    ref_points = []
    for i,y_point in enumerate(final_results_joint_to_find_on_line['y']):
        x_p = final_results_joint_to_find_on_line['x'][i]
        y_p = final_results_joint_to_find_on_line['y'][i]
        z_p = final_results_joint_to_find_on_line['z'][i]
    
        x_1 = final_results_joint_1['x'][i]
        y_1 = final_results_joint_1['y'][i]
        z_1 = final_results_joint_1['z'][i]
        x_2 = final_results_joint_2['x'][i]
        y_2 = final_results_joint_2['y'][i]
        z_2 = final_results_joint_2['z'][i]
        

        point = find_point(x_1,y_1,z_1,x_2,y_2,z_2,x_p,y_p,z_p)
       
    
        ref_points.append(point)
    return np.array(ref_points)

        
def plot_ref_points_w_body(final_results_dict,list_of_ref_points,desired_frame):
    plot_x_data = []
    plot_y_data = []
    plot_z_data = []
    plot_body = []
    for key in final_results_dict:
        # plot_x_data.append(final_results_dict[key]['x'][desired_frame])
        # plot_y_data.append(final_results_dict[key]['y'][desired_frame])
        # plot_z_data.append(final_results_dict[key]['z'][desired_frame])
        if key[-3:] == 'hip':
            plot_body.append(4)
            plot_x_data.append(final_results_dict[key]['x'][desired_frame])
            plot_y_data.append(final_results_dict[key]['y'][desired_frame])
            plot_z_data.append(final_results_dict[key]['z'][desired_frame])
        elif key[-3:] == 'der':
            plot_body.append(3)

            plot_x_data.append(final_results_dict[key]['x'][desired_frame])
            plot_y_data.append(final_results_dict[key]['y'][desired_frame])
            plot_z_data.append(final_results_dict[key]['z'][desired_frame])
        elif key[-3:] == 'ose':
            plot_body.append(2)
            plot_x_data.append(final_results_dict[key]['x'][desired_frame])
            plot_y_data.append(final_results_dict[key]['y'][desired_frame])
            plot_z_data.append(final_results_dict[key]['z'][desired_frame])
        elif key[-3:] == 'bow':
            plot_body.append(1)
            plot_x_data.append(final_results_dict[key]['x'][desired_frame])
            plot_y_data.append(final_results_dict[key]['y'][desired_frame])
            plot_z_data.append(final_results_dict[key]['z'][desired_frame])
       
    for ref_point in list_of_ref_points:
        plot_x_data.append(ref_point[desired_frame][0])
        plot_y_data.append(ref_point[desired_frame][1])
        plot_z_data.append(ref_point[desired_frame][2])
        plot_body.append(5)
    
    fig = go.Figure(data =[go.Scatter3d(x = plot_x_data,
                                    y = plot_y_data,
                                    z = plot_z_data,
                                    mode ='markers',
                                    marker = dict(
                                        size = 12,
                                        color = plot_body,
                                        colorscale ='Viridis',
                                        opacity = 0.8))])
    fig.layout.update(showlegend=True)
    fig.show()




def get_joint_angles(v1,v2):
    v1mag = np.sqrt(v1[0,:] * v1[0,:] + v1[1,:] * v1[1,:] + v1[2,:]  * v1[2,:] )
    v1norm = [v1[0,:] / v1mag, v1[1,:] / v1mag, v1[2,:] / v1mag]

    v2mag = np.sqrt(v2[0,:] * v2[0,:] + v2[1,:] * v2[1,:] + v2[2,:]  * v2[2,:] )
    v2norm = [v2[0,:] / v2mag, v2[1,:] / v2mag, v2[2,:] / v2mag]

    res = (v1norm[0] * v2norm[0]) + (v1norm[1] * v2norm[1]) + (v1norm[2] * v2norm[2])

    angle_rad = np.arccos(res)
    full_angle_deg = np.rad2deg(angle_rad)

    # for xy plane
    # xy_vec_a = np.array([v1[0],v1[1]])
    # xy_vec_b = np.array([v2[0],v2[1]])
    
    # xz_vec_a = np.array([v1[0],v1[2]])
    # xz_vec_b = np.array([v2[0],v2[2]])

    # yz_vec_a = np.array([v1[1],v1[2]])
    # yz_vec_b = np.array([v2[1],v2[2]])



    return full_angle_deg

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
    upper_arm_scores=[]
    lower_arm_scores = []
    wrist_scores = []
    neck_scores = []
    trunk_scores = []
    leg_scores = []
    for i,frame in enumerate(full_pose_dict['neck']):
        frame_angle_dict = {}
        for key in full_pose_dict:
            if key == 'l_upper_arm' or key == 'r_upper_arm':
                frame_angle_dict[key] = (full_pose_dict[key][0][i],full_pose_dict[key][1][i])
            else:
                frame_angle_dict[key] = full_pose_dict[key][i]
        reba_score = RebaScoreMIT(frame_angle_dict)
        score_a, partial_a = reba_score.compute_score_a()
        score_b, partial_b = reba_score.compute_score_b()
        upper_arm_score, lower_arm_score, wrist_score = partial_b
        neck_score, trunk_score, leg_score = partial_a
        upper_arm_scores.append(upper_arm_score)
        lower_arm_scores.append(lower_arm_score)
        wrist_scores.append(wrist_score)
        neck_scores.append(neck_score)
        trunk_scores.append(trunk_score)
        leg_scores.append(leg_score)
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
    return a_results,b_results,c_results,reba_class,upper_arm_scores,lower_arm_scores,wrist_scores,neck_scores,trunk_scores,leg_scores



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

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])