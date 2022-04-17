# Script to have Dataset class to use dataset loader

import cv2
import torch.utils.data as data 
import torch 
import numpy as np 
import h5py
import pickle 
import os 
import glob 

from os.path import join

class DynamicsDataset(data.Dataset):
    # Dataset that returns obs, obs_next, action pairs 

    def __init__(self, root): 
        self.root = root 

        with open(join(root, f'pos_pairs_1.pkl'), 'rb') as f:
            self.pos_pairs = pickle.load(f)

        images_folder = join(root, 'images')
        image_names = glob.glob(join(images_folder, 'frame*'))
        self.image_names = sorted(image_names)

        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images']

    # Returns image id to frame id which is going to be the id used for images
    def _img_to_id (self, img_name):
        frame_id = img_name.split('_')[-1] # Will have the frame_id.png
        frame_id = int(frame_id.split('.')[0])

        return frame_id

    # def _get_image(self, path): 
    #     img = cv2.imread(path) # Shape: 480, 640, 3
    #     img = np.transpose(img, (2,0,1))
    #     img = img.astype('float32') / 255
    #     img = (img - 0.5) / 0.5 # Normalize 

    #     return torch.FloatTensor(img)

    def _get_image(self, index):
        img = self.images[index]
        img = img.astype('float32') / 255 # NOTE: Why do we save it as ints then?
        img = (img - 0.5) / 0.5 

        return torch.FloatTensor(img)

    def _get_image_by_path(self, path):
        return self._get_image(self._img_to_id(path))

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index): 
        obs_file, obs_next_file, action = self.pos_pairs[index]
        obs = self._get_image_by_path(obs_file)
        obs_next = self._get_image_by_path(obs_next_file)

        # NOTE: (action - self.mean) / self.std var burda!! 

        return obs, obs_next, torch.FloatTensor(action)

if __name__ == '__main__':
    train_dset = DynamicsDataset(root='data/28012018_111425')
    train_loader = data.DataLoader(train_dset, batch_size=64, shuffle=True)
    batch = next(iter(train_loader))

    print('batch.shape: {}'.format(len(batch)))
    print(batch[0].shape, batch[1].shape, batch[2].shape)