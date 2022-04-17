#!/usr/bin/env python

import cv2
import cv_bridge
import rospy
import signal
import os
import shutil

from datetime import datetime

# ROS Image message
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, LaserScan, Joy # Image data will be taken - cv2 bridge can also be used to invert these but for now 
from ackermann_msgs.msg import AckermannDriveStamped # Steering angle and speed will be taken - header will also be saved! 

import pickle

class DataSaver:
    def __init__(self, node_name, ackermann_topic_name, video_topic_name, scan_topic_name):
        rospy.init_node(node_name, disable_signals=True)

        rospy.loginfo('inside the class')

        # Create a directory to save all the data
        now = datetime.now()
        time_str = now.strftime('%d%m%Y_%H%M%S')
        curr_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(curr_path, 'data_{}'.format(time_str))
        os.mkdir(self.data_dir)

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()

        # Create a directory to save all the images
        self.img_dir_path = os.path.join(self.data_dir, 'video_{}'.format(time_str)) # there will be multiple directories created all the time 
        os.mkdir(self.img_dir_path)
        self.video_fps = 40

        # These arrays will hold (stamp, <wanted data for each topic>)
        self.ackermann_msgs = [] # For ackermann we will need steering angle and speed
        self.scan_msgs = [] # For scan msgs we will need the ranges
        self.img_names = [] 
        self.manual_moves = [] # [[beginning_time, ending_time], [beginning_time, ending_time]....]
        self.dump_msgs = False
        self.stop_signal = False
        self.manual_move_started = False

        # Initialize ROS listeners
        ackermann_sub = rospy.Subscriber(ackermann_topic_name, AckermannDriveStamped, self.ackermann_callback)
        image_sub = rospy.Subscriber(video_topic_name, Image, self.video_callback)
        scan_sub = rospy.Subscriber(scan_topic_name, LaserScan, self.scan_callback)
        stop_sub = rospy.Subscriber('/car/stop', Bool, self.stopping_callback)
        joy_sub = rospy.Subscriber('/car/teleop/joy', Joy, self.joystick_callback) # This is for when we move the car manually 

        signal.signal(signal.SIGINT, self.end_signal_handler)

        rospy.spin()

    def ackermann_callback(self, data):
        if not self.stop_signal:
            stamp = rospy.get_rostime()
            stamp = float('{}.{:09d}'.format(stamp.secs, stamp.nsecs))
            steering_angle = data.drive.steering_angle
            speed = data.drive.speed
            self.ackermann_msgs.append((stamp, steering_angle, speed))

    def video_callback(self, data):
        if not self.stop_signal:
            stamp = rospy.get_rostime()
        
            cv2_img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        
            img_name = float('{}.{:09d}'.format(stamp.secs, stamp.nsecs))
            img_file_name = os.path.join(self.img_dir_path, '{}.jpg'.format(img_name))
            self.img_names.append(img_name) # Is used for loading data
            cv2.imwrite(img_file_name, cv2_img)

    def scan_callback(self, data):
        if not self.stop_signal:
            stamp = rospy.get_rostime()
            scan_data = data.ranges
            self.scan_msgs.append((stamp, scan_data))

    def joystick_callback(self, data):
        if data.buttons[2] == 1 and not self.manual_move_started: # circle button will indicate the starting time
            now = datetime.now()
            time_str = now.strftime('%d%m%Y_%H%M%S')
            self.manual_moves.append([time_str])
            rospy.loginfo('Manual move started at: {}, manuel_moves: {}'.format(time_str, self.manual_moves))
            self.manual_move_started = True

        elif data.buttons[0] == 1 and self.manual_move_started:
            now = datetime.now()
            time_str = now.strftime('%d%m%Y_%H%M%S')
            self.manual_moves[-1].append(time_str)
            rospy.loginfo('Manual move ended at: {}, manual_moves: {}'.format(time_str, self.manual_moves))
            self.manual_move_started = False

    def stopping_callback(self, data):
        self.stop_signal = data.data
        if self.stop_signal and self.dump_msgs: # If sop msg is true then dump the msgs
            
            now = datetime.now()
            time_str = now.strftime('%d%m%Y_%H%M%S')
            rospy.loginfo('dumping in {}'.format(time_str))
            with open('{}/ackr_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
                pickle.dump(self.ackermann_msgs, pkl, pickle.HIGHEST_PROTOCOL)
            with open('{}/img_names_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
                pickle.dump(self.img_names, pkl, pickle.HIGHEST_PROTOCOL)
            with open('{}/scan_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
                pickle.dump(self.scan_msgs, pkl, pickle.HIGHEST_PROTOCOL)
            with open('{}/manual_moves_{}.txt'.format(self.data_dir, time_str), 'w') as f:
                f.write(str(self.manual_moves))

            self.ackermann_msgs = [] # For ackermann we will need steering angle and speed
            self.scan_msgs = [] # For scan msgs we will need the ranges
            self.img_names = []
            self.manual_moves = []

            # Convert images to video, remove the last video directory and create a new one
            # with a new timestamp

            # First, create the new directory if the converting doesn't end before starting moving again
            old_img_dir_path = str(self.img_dir_path)
            # Create a new directory
            self.img_dir_path = os.path.join(self.data_dir, 'video_{}'.format(time_str)) # there will be multiple directories created all the time
            os.mkdir(self.img_dir_path)

            print('old_img_dir_path: {}, self.img_dir_path: {}'.format(old_img_dir_path, self.img_dir_path))

            video_name = '{}/video_{}.mp4'.format(self.data_dir, time_str) 
            os.system('ffmpeg -f image2 -r {} -i {}/%*.jpg -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
                self.video_fps, # fps
                old_img_dir_path,
                video_name
            ))
            # Remove the img directory
            shutil.rmtree(old_img_dir_path)

            self.dump_msgs = False

            after_dumping = datetime.now()
            time_spent = after_dumping - now
            rospy.loginfo('DUMPING DONE in {} minutes\n-------------'.format(time_spent.seconds / 60.))

        elif not self.stop_signal:

            self.dump_msgs = True

    def end_signal_handler(self, signum, frame):
        rospy.loginfo('in end_signal_handler!')
        now = datetime.now()
        time_str = now.strftime('%d%m%Y_%H%M%S')
        with open('{}/ackr_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
            pickle.dump(self.ackermann_msgs, pkl, pickle.HIGHEST_PROTOCOL)
        with open('{}/img_names_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
            pickle.dump(self.img_names, pkl, pickle.HIGHEST_PROTOCOL)
        with open('{}/scan_msgs_{}.pickle'.format(self.data_dir, time_str), 'wb') as pkl:
            pickle.dump(self.scan_msgs, pkl, pickle.HIGHEST_PROTOCOL)
        with open('{}/manual_moves_{}.txt'.format(self.data_dir, time_str), 'w') as f:
            f.write(str(self.manual_moves))

        video_name = '{}/video_{}.mp4'.format(self.data_dir, time_str)
        os.system('ffmpeg -f image2 -r {} -i {}/%*.jpg -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
            self.video_fps, # fps
            self.img_dir_path,
            video_name
        ))
        # Remove the img directory
        shutil.rmtree(self.img_dir_path)

        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)


if __name__ == '__main__':
    
    data_saver = DataSaver(
        node_name = 'arya_saver',
        ackermann_topic_name = '/car/mux/ackermann_cmd_mux/input/teleop',
        video_topic_name = '/car/csi_cam_0/image_raw',
        scan_topic_name = '/car/scan'
    )
