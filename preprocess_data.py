# Script to split the videos in current timesteps to runs in 11 frames, similar to cfm's approach
# Each run will have 11 png files and actions.npy for each frame
# Multiple runs will be created for that particular timestepped data bunch

import argparse 
import cv2
import imageio
import random
import sys
import os
from os.path import join, dirname, basename 
import glob
import pickle 
import numpy as np
import h5py
import torch 

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import default_loader as loader 

import utils

interval_ts = 1 # number of timesteps bw current image / next image pairs 

class DataProcesser:
    # root: name of the data folder to get the video and pickle files
    def __init__(self, root):
        # Create a directory named images under root to dump the images
        # self.root = root 
        # NOTE: Eger bi noktada birden fazla dosyayi ayni anda halletmeye calisirsak diye 
        # bilgin olsun: glob.glob tabisi bi liste veriyo bize 
        self.video_path = glob.glob(join(root, 'video_*'))[0]
        self.ackr_file_path = glob.glob(join(root, 'ackr_msgs_*'))[0]
        self.img_names_file_path = glob.glob(join(root, 'img_names_*'))[0]
        self.manual_moves_file_path = glob.glob(join(root, 'manual_moves_*'))[0]
        self.images_folder = join(root, 'images')
        self.actions_file_path = join(root, 'actions.npy') # This file will be created if not
        self.root = root

        # print('video: {}, actions: {}, img_names: {}, manual_moves: {}'.format(
        #     self.video_path, self.ackr_file_path, self.img_names_file_path,
        #     self.manual_moves_file_path
        # ))

        # TODO: split train/test dataset at some point
        # TODO: Check manual moves!!

        # Create the directory to dump all the images
        # And dump video to images if so
        if not os.path.isdir(self.images_folder):
            self.dump_video_to_images()
            os.mkdir(self.images_folder)

        # Match actions with frames
        if not os.path.isfile(self.actions_file_path):
            self.match_actions_with_frames()

        # Create positive pairs and plot them if desired
        # if not os.path.isfile(join(root, 'pos_pairs_1.pkl')): # TODO: 1'i degistirince...
        self.create_pos_pairs(plot_pairs=True)

        # Load the images into hdf5 dataset for faster training
        if not os.path.isfile(join(self.root, 'images.hdf5')):
            self.load_images()


    def dump_video_to_images(self):
        # Matching of frames and actions will be done when we are creating
        # the positive and negative pairs

        # Convert the video into image sequences and name the images
        # according to the img_names array
        vidcap = cv2.VideoCapture(self.video_path)
        success, image = vidcap.read()
        frame_id = 0
        while success: # The matching 
            cv2.imwrite('{}.png'.format(join(self.images_folder, 'frame_{}'.format(str(frame_id).zfill(5)))),
                        image)     # save frame as JPEG file      
            success,image = vidcap.read()
            # print('Read a new frame: ', success)
            frame_id += 1

        print('dumping finished')

    # Method to match saved ackermann messages with the dumped frames
    def match_actions_with_frames(self):
        # Get the img name and ackermann message pickle files
        with open(self.ackr_file_path, 'rb') as pkl_file:
            ackermann_data = np.array(pickle.load(pkl_file))

        with open(self.img_names_file_path, 'rb') as pkl_file:
            img_names = np.array(pickle.load(pkl_file))

        # Traverse through image names and ackermann_data and find the matching
        # ones
        # actions will hold: [frame_id, steering_angle, linear_speed]
        self.actions = np.ones((len(img_names), 3)) * -1 # First all values will be -1

        ackermann_is_more_freq = ackermann_data.shape[0] > img_names.shape[0]

        i = 0
        j = 0
        while j < len(ackermann_data) and i < len(img_names):

            i,j = self.match_indices(img_names, ackermann_data, i, j)
            if i < len(img_names) and j < len(ackermann_data):

                self.actions[i,:] = np.array([i, ackermann_data[j,1], ackermann_data[j,2]])[:]

        # Dump the actions as npy file
        np.save(self.actions_file_path, self.actions)

    def match_indices(self, img_names, ackr_data, i, j):
        # Method that returns the corresponding indices for matching values
        # in given arrays
        # It will find the closest value in arr2 for every value in arr1 (other way around
        # if values in arr2 are in general larger)
        # i: beginning index of img_names
        # j: beginning index of ackr_data

        if img_names[i] < ackr_data[j,0]:
            while i < len(img_names) and img_names[i] < ackr_data[j,0]:
                i += 1
            return i,j 

        else: # ackr_data[j,0] < img_names[i]
            while j < len(ackr_data) and ackr_data[j,0] < img_names[i]:
                j += 1
            return i,j

    # Pair the images as positive and negative and
    # save them in a pickle file including actions between those frames
    def create_pos_pairs(self, plot_pairs=False): 
        # Traverse through the images that were dumped and actions
        image_names = glob.glob(join(self.images_folder, 'frame*'))
        image_names = sorted(image_names)

        # Get the actions that had an action
        actions = np.load(self.actions_file_path)

        # TODO: Only the adjacent frame is added - interval_ts is 1 and everything is
        # implemented that way
        pos_pairs = []

        for image_name in image_names:
            frame_id = image_name.split('_')[-1] # Will have the frame_id.png
            frame_id = int(frame_id.split('.')[0]) # Will have the frame_id as str
            
            if frame_id+1 == len(image_names):
                break

            if frame_id < actions.shape[0] and actions[frame_id,0] != -1:
                pos_pairs.append((image_name, image_names[frame_id+1], actions[frame_id,1:]))

        with open(join(self.root, 'pos_pairs_1.pkl'), 'wb') as f:
            pickle.dump(pos_pairs, f)

        # print('pos_pairs[:100]: {}'.format(pos_pairs[:100]))
        # print(image_names[:100])

        # TODO: Create a DynamicDataset adjacent to get the images with a data loader
        # This will return all the images as a torch and will help the dumping 
        # but for now we will get random images from pos_pairs only
        # Get random images
        if plot_pairs:
            n_images = 16
            rand_pos_pairs = random.choices(pos_pairs, k=n_images)
            # print(rand_pos_pairs[:4])

            imgs = np.zeros((n_images*2, 480,640,3))
            for i,pos_pair in enumerate(rand_pos_pairs):
                img, img_next = cv2.imread(pos_pair[0]), cv2.imread(pos_pair[1])
                # print('img.shape: {}, img_next.shape: {}'.format(img.shape, img_next.shape))
                action = pos_pair[2]
                img = utils.add_arrow(img, action)

                imgs[2*i,:] = img[:]
                imgs[2*i+1,:] = img_next[:]

            # print('first channel mean and std: {},{}'.format(imgs[:,0,:,:].mean(), imgs[:,0,:,:].std()))
            # print('second channel mean and std: {},{}'.format(imgs[:,1,:,:].mean(), imgs[:,0,:,:].std()))
            # print('third channel mean and std: {},{}'.format(imgs[:,2,:,:].mean(), imgs[:,0,:,:].std()))
            imgs /= 255.
            imgs = torch.FloatTensor(imgs).permute(0,3,1,2) # (n_image,3,480,640)
            save_image(imgs, join(self.root, 'pos_pair_exs_1.png'), nrow=8)

    def load_images(self):
        # Preloads images into an hdf5 dataset for faster access and training
        # Images are not normalized and then they are transformed into integers
        image_names = glob.glob(join(self.images_folder, 'frame*'))
        image_names = sorted(image_names)

        transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.CenterCrop((480,640)),
            transforms.ToTensor(),
            transforms.Normalize((150,150,150), (60,60,60)) # NOTE: Might want to change these, these are found by few images
        ])

        # NOTE: Add all these transforms as notes to Lerrel
        dset = h5py.File(join(self.root, 'images.hdf5'), 'x')
        dset.create_dataset('images', (len(image_names),3,480,640), 'uint8')
        for i, img in enumerate(tqdm(image_names)):
            img = transform(loader(img)) # Load the image
            img = img.numpy() * 0.5 + 0.5
            img += 255 
            img = img.astype(np.uint8)
            dset['images'][i] = img

        print('dataset loaded')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/28012018_111425', help='path to data action and video')
    args = parser.parse_args()

    dp = DataProcesser(args.root)