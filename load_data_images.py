import pickle
import numpy as np
import rospy
import cv_bridge
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Script to load the saved pickle files, match ackermann commands and convert them to images and human readable ackermann commands (json file for that period)

ackermann_pickle_file = 'ackermann_msgs.pickle' # TODO: These should probs be set according to time and day of the save
video_pickle_file = 'img_names.pickle'
scan_pickle_file = 'scan_msgs.pickle'

# Get ackermann values
with open(ackermann_pickle_file, 'rb') as pkl_file:
    ackermann_data = np.array(pickle.load(pkl_file))

# Get video values
with open(video_pickle_file, 'rb') as pkl_file:
    video_data = np.array(pickle.load(pkl_file))

# Get laser scan values
with open(scan_pickle_file, 'rb') as pkl_file:
    scan_data = np.array(pickle.load(pkl_file))

# Create the angle array for scan visualization
angle_min = -3.14159274101
angle_max = 3.14159274101
angle_increment = 0.00872664619237
angle_arr = np.arange(angle_min, angle_max, angle_increment)[:-1] # Don't include the angle_max

# roll both angle_arr and scan_data to make the data start from the 0 angle
angle_arr = np.roll(angle_arr, 540)
scan_data = np.roll(scan_data, 540, axis=1)

print('video_data.len: {}'.format(len(video_data)))
print('ackermann_data.len: {}'.format(len(ackermann_data)))
print('scan_data.len: {}'.format(len(scan_data)))

curr_ackermann = {
    'timestamps': [],
    'steering_angles': [],
    'speeds': []
}

FIXED_TIMESTAMP = 100 # Only 50 values will be shown in steering_angles and speeds

# Create the directory to save the images with the ros time name
curr_file_path = os.path.dirname(os.path.abspath(__file__))
curr_dir_name = str(video_data[0])
dir_path = os.path.join(curr_file_path, curr_dir_name) 
os.mkdir(dir_path)

# Create the matplotlib plt
f, ((a0, a1, a2), (a3, _, _)) = plt.subplots(2, 3, figsize=(23,16), gridspec_kw={'width_ratios': [2,1,1]})

# Traverse through the video data and print the times of the messages
i = 0 # video data index
j = 0 # ackermann data index
k = 0 # scan data index

img_plt = None
steer_plt = None
speed_plt = None
scan_plt = None

while j < len(ackermann_data) and i < len(video_data) and k < len(scan_data):
    scan_data_time = float('{}.{:09d}'.format(scan_data[k][0].secs, scan_data[k][0].nsecs)) # I needed to save scan timestamp as a rospy time
    if ackermann_data[j][0] > video_data[i]: # scan_data is the least frequent one
        print('ackermann_data[{}][0]: {} ----------- scan_data[{}][0]: {} ----------  video_data[{}][0]: {}'.format(
            j, ackermann_data[j][0], k, scan_data_time, i, video_data[i]))
       
        # Read the image with 'video/video_data[i].jpg'
        video_img = mpimg.imread('video/{}.jpg'.format(str(video_data[i])))
        # Save your OpenCV2 image as a jpeg 
        # Plot image, steering angle and speed side by side using matplotlib
        if img_plt is None:
            img_plt = a0.imshow(video_img)
            a0.set_title('Taken Image')
        else:
            img_plt.set_data(video_img)

        if steer_plt is None:
            steer_plt, = a1.plot(curr_ackermann['timestamps'], curr_ackermann['steering_angles'], 'r-')
            a1.set_title('Steering Angles in Radian'), a1.set_ylim([-0.31, 0.31])
        else:
            steer_plt.set_data(curr_ackermann['timestamps'], curr_ackermann['steering_angles'])
            a1.relim()
            a1.autoscale_view(True, True, True)

        if speed_plt is None:
            speed_plt, = a2.plot(curr_ackermann['timestamps'], curr_ackermann['speeds'], 'b-')
            a2.set_title('Linear Speed'), a2.set_ylim([-2.0, 2.0])
        else:
            speed_plt.set_data(curr_ackermann['timestamps'], curr_ackermann['speeds'])
            a2.relim()
            a2.autoscale_view(True, True, True)

        scan_x = np.multiply(scan_data[k,1], np.cos(angle_arr))
        scan_y = np.multiply(scan_data[k,1], np.sin(angle_arr))
        if scan_plt is None:
            scan_plt, = a3.plot(scan_x, scan_y, 'o')
            a3.arrow(0,0,0,1.5, ec='red', width=0.05, head_width=0.2)
            a3.set_xlim([-15,15]), a3.set_ylim([-15,15]), a3.set_title('Scan Values')
        else:
            scan_plt.set_data(scan_x, scan_y)
    
        plt.savefig('{}/{}.jpeg'.format(dir_path, str(video_data[i])))
        i += 1
    
    if video_data[i] > scan_data_time:
        k += 1

    if len(curr_ackermann['timestamps']) >= FIXED_TIMESTAMP:
        curr_ackermann['timestamps'].pop(0)
        curr_ackermann['steering_angles'].pop(0)
        curr_ackermann['speeds'].pop(0)
        
    curr_ackermann['timestamps'].append(ackermann_data[j][0])
    curr_ackermann['steering_angles'].append(float(ackermann_data[j][1]))
    curr_ackermann['speeds'].append(float(ackermann_data[j][2]))

    print('curr_ackermann[timestamps][0]: {}, steering_angles[0]: {}, speeds[0]: {}, len: {},{},{}'.format(
        curr_ackermann['timestamps'][0], curr_ackermann['steering_angles'][0], curr_ackermann['speeds'][0],
        len(curr_ackermann['timestamps']), len(curr_ackermann['steering_angles']), len(curr_ackermann['speeds'])))

    j += 1
    
