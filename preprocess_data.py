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

from animation import AnimationWrapper

import utils

interval_ts = 20 # number of timesteps bw current image / next image pairs 

class DataProcesser:
    # root: name of the data folder to get the video and pickle files
    def __init__(self, roots, sec_interval=5, poly_max_deg=5,
                       dump_action_video=False, plot_pairs=False):

        # Do the whole thing for each given root and then dataset loader will handle
        # multiple roots inside
        fps = 20
        self.sec_interval = sec_interval 
        self.poly_max_deg = poly_max_deg
        self.frame_interval = int(fps * sec_interval)
        self.dump_action_video = dump_action_video
        self.plot_pairs = plot_pairs

        for root in roots:
            print('current root: {}'.format(root))

            self.video_path = glob.glob(join(root, 'video_*'))[0]
            self.ackr_file_path = glob.glob(join(root, 'ackr_msgs_*'))[0]
            self.img_names_file_path = glob.glob(join(root, 'img_names_*'))[0]
            if len(glob.glob(join(root, 'manual_moves_*'))) > 0:
                self.manual_moves_file_path = glob.glob(join(root, 'manual_moves_*'))[0]
            else:
                self.manual_moves_file_path = None
            self.images_folder = join(root, 'images')
            self.actions_file_path = join(root, 'actions.npy') # This file will be created if not
            self.root = root

            # Dump images to video 
            if not os.path.isdir(self.images_folder):
                os.mkdir(self.images_folder)
                self.dump_video_to_images()

            if not os.path.isfile(self.actions_file_path):
                self.match_actions_with_frames() # Will dump actions file with -1 for non matched frames

            # if not os.path.isfile(join(self.root, 'pos_pairs.pkl')): # TODO: delete pos_pairs and images.hdf5 before doing all this 
            # self.create_pos_pairs()
            self.create_pos_pairs_guided()

            # if not os.path.isfile(join(self.root, 'images.hdf5')):
            #     self.load_images()

    def dump_video_to_images(self):
        # Matching of frames and actions will be done when we are creating
        # the positive and negative pairs

        # Convert the video into image sequences and name the images
        # according to the img_names array
        vidcap = cv2.VideoCapture(self.video_path)
        success, image = vidcap.read()
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the wanted frames 
        # self.frames_to_take = self.find_changing_indices()

        frame_id = 0
        # i = 0
        print('dumping video in {}'.format(self.root))
        pbar = tqdm(total = frame_count)
        while success: # The matching 
            pbar.update(1)
            cv2.imwrite('{}.png'.format(join(self.images_folder, 'frame_{}'.format(str(frame_id).zfill(5)))), image)
            success,image = vidcap.read()
            frame_id += 1

        print('dumping finished in {}'.format(self.root))

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
        while i < len(img_names) and j < len(ackermann_data):

            i,j = self.match_indices(img_names, ackermann_data, i, j)
            if i < len(img_names) and j < len(ackermann_data):

                # NOTE: Linear speed was initially saved negatively so we multiply it with -
                # TODO: You changed this part!! 
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

    # Method to find changing velocity commands by looking at the acceleration
    # of the velocity commands, if the acceleration is highest around that frame's
    # previous and next frame's acceleration, it means that the velocity comamnd has
    # changed
    def find_changing_indices(self):
        # Get the actions 
        actions = np.load(self.actions_file_path)

        # Traverse actions and if the change in command is highest then it means
        # that there is a change in the velocity command
        frames_to_take = []
        curr_max_acc = 0
        for i in range(len(actions)-1):
            if (actions[i-1:i+2,0] != -1).all():
                # It's enough to only check the change in linear speed
                curr_acc = abs(actions[i,2]-actions[i+1,2])
                prev_acc = abs(actions[i-1,2]-actions[i,2]) if i > 0 else 0
                if curr_acc > prev_acc:
                    frames_to_take.append(i+1)

        print('frames_to_take: {}'.format(frames_to_take[:50]))

        return frames_to_take

    # This method is for creating the positive/negative pairs when the data was collected
    # when the car wasn't moving randomly but it was moved by me
    def create_pos_pairs_guided(self):
        image_names = glob.glob(join(self.images_folder, 'frame*'))
        image_names = sorted(image_names)
        
        # Get actions - not matched actions will have -1 as 
        actions = np.load(self.actions_file_path)

        print(f'len(image_names): {len(image_names)}, len(actions): {len(actions)}')

        non_zero_actions = actions[actions[:,0] != -1]
        print(f'len(non_zero_actions): {len(non_zero_actions)}')
        # print('np.where(actions[:,0] != -1): {}'.format(np.where(actions[:,0] != -1)))

        exist_act_indices = np.where(actions[:,0] != -1)[0]
        print('exist_act_indices.shape: {}'.format(exist_act_indices.shape))
        pos_pairs = []
        for i in range(len(exist_act_indices)-self.frame_interval):
            fir_img_id = exist_act_indices[i]
            sec_img_id = exist_act_indices[i+self.frame_interval]
            # print('fir_img_id: {}, sec_img_id: {}'.format(fir_img_id, sec_img_id))
            if sec_img_id >= len(image_names):
                break

            curr_actions = actions[exist_act_indices[i:i+self.frame_interval]]
            # print('curr_actions: {}'.format(curr_actions[:,1:]))
            mean_curr = np.mean(curr_actions[:,1:], axis=0)
            # print('mean_curr: {}'.format(mean_curr))

            pos_pairs.append((
                image_names[fir_img_id],
                image_names[sec_img_id],
                mean_curr[:]
            ))

        with open(join(
            self.root,
            f'pos_pairs_sec_{self.sec_interval}_mean.pkl'), 'wb') as f:
            pickle.dump(pos_pairs, f) # all pos_pairs will be appended to each other in dataset loader

        # This will return all the images as a torch and will help the dumping 
        # but for now we will get random images from pos_pairs only
        # Get random images
        if self.plot_pairs:
            n_images = 16
            rand_pos_pairs = random.choices(pos_pairs, k=n_images)
            # print(rand_pos_pairs[:4])

            imgs = np.zeros((n_images*2, 480,640,3))
            for i,pos_pair in enumerate(rand_pos_pairs):
                img, img_next = cv2.imread(pos_pair[0]), cv2.imread(pos_pair[1])
                action = pos_pair[2]
                img = utils.add_arrow(img, action)

                imgs[2*i,:] = img[:]
                imgs[2*i+1,:] = img_next[:]

            imgs /= 255.
            imgs = torch.FloatTensor(imgs).permute(0,3,1,2) # (n_image,3,480,640)
            save_image(imgs, join(
                self.root,
                f'pos_pairs_sec_{self.sec_interval}_exs.png'), nrow=8)

    # Pair the images as positive and negative and
    # save them in a pickle file including actions between those frames
    def create_pos_pairs(self): 
        # Traverse through the images that were dumped and actions
        image_names = glob.glob(join(self.images_folder, 'frame*'))
        image_names = sorted(image_names)

        # Get the actions that had an action
        actions = np.load(self.actions_file_path)

        # Polynom includes set action coefficients for each action 
        polynoms = self.fit_model_to_actions(actions)
        print('polynoms.shape: {}, len(image_names): {}, self.frame_interval: {}'.format(polynoms.shape, len(image_names), self.frame_interval))

        pos_pairs = []

        poly_index = -1
        act_index = 0
        while poly_index < polynoms.shape[0]-1 and act_index < actions.shape[0]-1:
            # Find the action that does not -1 value in its first index (which means that there is a corresponding frame)
            if actions[act_index, 0] == -1:
                while actions[act_index,0] == -1:
                    act_index += 1
            else:
                act_index += 1
            poly_index += 1
            
            # Find the second image to add
            sec_img_index = act_index 
            dist_to_frame_interval = self.frame_interval
            while dist_to_frame_interval > 0:
                if actions[sec_img_index, 0] == -1:
                    while actions[sec_img_index, 0] == -1:
                        sec_img_index += 1
                else:
                    sec_img_index += 1
                dist_to_frame_interval -= 1

            pos_pairs.append((
                image_names[act_index],
                image_names[sec_img_index],
                polynoms[poly_index,:]
            ))

        print('len(pos_pairs): {}'.format(len(pos_pairs)))
        print(pos_pairs[:10])
        assert len(pos_pairs) == polynoms.shape[0]
        with open(join(
            self.root,
            f'pos_pairs_sec_{self.sec_interval}_deg_{self.poly_max_deg}.pkl'), 'wb') as f:
            pickle.dump(pos_pairs, f) # all pos_pairs will be appended to each other in dataset loader

        # This will return all the images as a torch and will help the dumping 
        # but for now we will get random images from pos_pairs only
        # Get random images
        if self.plot_pairs:
            n_images = 16
            rand_pos_pairs = random.choices(pos_pairs, k=n_images)
            # print(rand_pos_pairs[:4])

            imgs = np.zeros((n_images*2, 480,640,3))
            for i,pos_pair in enumerate(rand_pos_pairs):
                img, img_next = cv2.imread(pos_pair[0]), cv2.imread(pos_pair[1])
                # action = pos_pair[3]
                # img = utils.add_arrow(img, action)

                imgs[2*i,:] = img[:]
                imgs[2*i+1,:] = img_next[:]

            imgs /= 255.
            imgs = torch.FloatTensor(imgs).permute(0,3,1,2) # (n_image,3,480,640)
            save_image(imgs, join(
                self.root,
                f'pos_pairs_sec_{self.sec_interval}_deg_{self.poly_max_deg}_exs.png'), nrow=8)

    # frame_interval: interval for the action model to be extracted 
    # sec_interval: number of seconds used for the action model 
    # model to fit the action graph in this interval will be created
    def fit_model_to_actions(self, actions): 
        print('actions.shape: {}'.format(actions.shape))
        non_zero_actions = actions[ actions[:,0] != -1 ] # actions that have a corresponding frame 
        fps = 20
        # frame_interval = sec_interval * fps # We have saved the actions with 20 fps 
        num_frames = non_zero_actions.shape[0]
        total_frames = num_frames - self.frame_interval # The last 20 will be used in the last action
        
        if self.dump_action_video: 
            X = np.linspace(0, num_frames*1.0/fps, num_frames) # This will only be used for plotting
            Y_pred = np.zeros((total_frames, self.frame_interval)) # We will have Ys created 
            Y_act = np.zeros((total_frames, self.frame_interval))

        P = np.zeros((total_frames, 2*(self.poly_max_deg+1))) # max degree + 1 coefficients are needed (including linear one) 
        # 2: 0 for steering angle, 1 for linear speed
        for i in range(total_frames):
            x = np.linspace(0, self.sec_interval, self.frame_interval)
            y_steer = non_zero_actions[i:i+self.frame_interval,0]
            y_lin = non_zero_actions[i:i+self.frame_interval,1]

            p_steer = np.polyfit(x, y_steer, deg=self.poly_max_deg)
            p_lin = np.polyfit(x, y_lin, deg=self.poly_max_deg)

            P[i,:self.poly_max_deg+1] = p_steer[:]
            P[i,self.poly_max_deg+1:] = p_lin[:]

            if self.dump_action_video:
                # Get the Y with poly
                y_pred = self.poly(x, p_lin)
                Y_act[i,:] = y_lin[:] # Video will be with linear change
                Y_pred[i,:] = y_pred[:]

        # print('X: {}'.format(X.shape))

        if self.dump_action_video:
            AnimationWrapper(
                X = X,
                Y_pred = Y_pred,
                Y_act = Y_act,
                dump_dir = self.root,
                dump_file = f'action_anim_sec_{self.sec_interval}_deg_{self.poly_max_deg}.mp4',
                total_frames = total_frames,
                sec_interval = self.sec_interval,
            )

        return P

    def poly(self, x, p):
        n = len(p)-1
        return sum([ x ** (n-i) * p[i] for i in range(n+1) ])

    # This method is not used anymore
    def load_images(self):
        # Preloads images into an hdf5 dataset for faster access and training
        # Images are not normalized and then they are transformed into integers
        image_names = glob.glob(join(self.images_folder, 'frame*'))
        image_names = sorted(image_names)

        transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.CenterCrop((480,480)), # TODO: Burda 480,480 yap bunu
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # NOTE: Add all these transforms as notes to Lerrel
        dset = h5py.File(join(self.root, 'images.hdf5'), 'x')
        dset.create_dataset('images', (len(image_names),3,480,480), 'uint8')
        for i, img in enumerate(tqdm(image_names)):
            img = transform(loader(img)) # Load the image
            img = img.numpy() * 0.5 + 0.5
            img *= 255 
            img = img.astype(np.uint8)
            dset['images'][i] = img

        print('dataset loaded')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/28012018_111425', help='path to data action and video')
    
    roots = glob.glob('data/*')
    print('roots: {}'.format(roots))
    
    args = parser.parse_args()

    # roots = ['data/28012018_110126'] # TODO: change this
    dp = DataProcesser(roots, sec_interval=0.5, poly_max_deg=10,
                       dump_action_video=False, plot_pairs=True)