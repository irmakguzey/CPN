# Script to have Dataset class to use dataset loader

import cv2
import torch.utils.data as data 
import torch 
import numpy as np 
# from mpi4py import MPI # For multi processing part
import h5py
import pickle 
import os 
import glob 
from tqdm import tqdm

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import default_loader as loader 
from PIL import Image
from os.path import join

class DynamicsDataset(data.Dataset):
    # Dataset that returns obs, obs_next, action pairs 

    def __init__(self, roots, sec_interval,
                 input_height=480, gaussian_blur=True, jitter_strength=1.): 
        roots = sorted(roots)
        self.roots = roots

        self.pos_pairs = []
        self.images_dset = []

        # pos_pairs_file_name = f'pos_pairs_sec_{sec_interval}_deg_{poly_max_deg}.pkl'
        pos_pairs_file_name = f'pos_pairs_sec_{sec_interval}_mean.pkl'


        # TODO: Set these action means according to the means and stds of the batches 
        # self.action_mean = np.array([-5.0e-3, 1.6e-1, -3.0e-1, 1.6e-1, 4e+1, 5.9e+3, 2.3e-3, -5.2e-3,
        #                              4.5e-3, -2.4e-3, 1.1e-3, -1.1e-2]) # These values are taken by looking at the data
        # self.action_std = np.array([1.2e+2, 3.2e+2, 2.8e+2, 1.0e+2, 1.5e+1, 3.5e+3, 1.3, 3.3, 2.97,
        #                             1.1, 2.0e-1, 1.4e-1])

        # means when sec=2 poly=5
        # self.action_mean = np.array([-1.3e-2, 4.0e-2, -5.0e-2, 4.0e-2, 2.0e+1, 5.9e+3, 2.9e-4, -1.1e-3,
        #                              1.2e-3, 5.0e-4, -9.6e-4, -1.2e-2]) # These values are taken by looking at the data
        # self.action_std = np.array([3.9e+0, 1.9e+1, 3.4e+1, 2.5e+1, 7.0e+0, 3.5e+3, 4.2e-2, 2.1e-1, 3.7e-1,
        #                             2.8e-1, 1.0e-1, 1.4e-1])

        # mean and std for sec=5 deg=10
        # self.action_mean = np.array([
        #     -1.7e-5, 5.0e-4, -6.0e-3, 3.7e-2, -1.4e-1,
        #     3.5e-1, -4.5e-1, 3.0e-1, -6.3e-2, 2.0e+1,
        #     5.8e+3, 7.2e-7, -1.8e-5, 2.1e-4, -1.1e-3,
        #     4.3e-3, -1.0e-2, 1.5e-2, -1.2e-2, 5.4e-3,
        #     -6.0e-4, -9.8e-3
        # ])
        # self.action_std = np.array([
        #     1.2e-2, 3.0e-1, 3.2e+0, 1.9e+1, 6.8e+1, 1.5e+2,
        #     2.1e+2, 1.7e+2, 7.0e+1, 1.2e+1, 3.5e+3, 1.4e-4,
        #     3.6e-3, 3.8e-2, 2.3e-1, 8.0e-1, 1.8e+0, 2.4e+0,
        #     1.9e+0, 7.6e-1, 1.4e-1, 1.5e-1
        # ])

        # sec=3, deg=10 means/stds
        # self.action_mean = np.array([
        #     3.8e-6, -2.1e-4, 3.3e-3, -2.5e-2, 8.6e-2,
        #     -2.3e-1, 3.4e-1, -2.7e-1, 1.1e-1, 2.0e+1,
        #     5.9e+3, 4.9e-8, 4.3e-8, -1.7e-5, 1.4e-4,
        #     -8.8e-4, 2.0e-3, -3.7e-3, 3.8e-3, -1.8e-3,
        #     -2.0e-4, -1.0e-2
        # ])
        # self.action_std = np.array([
        #     1.3e-2, 3.4e-1, 3.6e+0, 2.1e+1, 7.7e+1, 1.7e+2,
        #     2.3e+2, 1.8e+2, 7.6e+1, 1.3e+1, 3.5e+3, 1.5e-4,
        #     3.5e-3, 3.8e-2, 2.2e-1, 8.0e-1, 1.8e+0, 2.4e+0,
        #     1.9e+0, 7.6e-1, 1.4e-1, 1.5e-1
        # ])

        # manually guided std and means
        self.action_mean = np.array([0, 0.35])
        self.action_std = np.array([0.09, 0.13])

        for root in self.roots:
            with open(join(root, pos_pairs_file_name), 'rb') as f:
                self.pos_pairs += pickle.load(f) # pos pairs is all indexed so it can be used

        # SimCLR transforms
        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        self.transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.RandomResizedCrop(size=self.input_height), 
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply([self.color_jitter], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * self.input_height + 1),
                                    sigma=(0.1, 2.0)),
            transforms.CenterCrop((480,480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _get_image(self, path): 

        img = self.transform(loader(path))

        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index): 
        obs_file, obs_next_file, action = self.pos_pairs[index]

        obs = self._get_image(obs_file)
        obs_next = self._get_image(obs_next_file)

        # TODO: Normalize the actions
        action = (action - self.action_mean) / self.action_std

        return obs, obs_next, torch.FloatTensor(action)

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in animation class 

class SimCLRDataset(data.Dataset):
    # Pretty similar to actual dataset but only difference there are more transforms
    # that are mentioned in the SimCLR paper
    def __init__(self, roots, sec_interval,
                 input_height=480, gaussian_blur=True, jitter_strength=1., use_eval=False):
        roots = sorted(roots)
        self.roots = roots

        self.pos_pairs = []
        self.images_dset = []

        pos_pairs_file_name = f'pos_pairs_sec_{sec_interval}_mean.pkl'

        # manually guided std and means
        self.action_mean = np.array([0, 0.35])
        self.action_std = np.array([0.09, 0.13])

        for root in self.roots:
            with open(join(root, pos_pairs_file_name), 'rb') as f:
                self.pos_pairs += pickle.load(f) # pos pairs is all indexed so it can be used

        # SimCLR transforms
        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        self.use_eval = use_eval # This boolean is used whether to use eval or train_transform
        # if set to false self.train_transform is used

        # Since there is a randomness - this is where the difference happens between two 
        # transformed embeddings
        self.train_transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.RandomResizedCrop(size=self.input_height), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * self.input_height + 1),
                                    sigma=(0.1, 2.0)),
            transforms.CenterCrop((480,480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.CenterCrop((480,480)), # TODO: Burda 480,480 yap bunu
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    
    def __len__(self):
        return len(self.pos_pairs)

    def _get_image(self, path): 

        if self.use_eval:
            img = self.eval_transform(loader(path))
        else:
            img = self.train_transform(loader(path))

        return torch.FloatTensor(img)

    def __getitem__(self, index): 
        obs_file, _, action = self.pos_pairs[index]

        obs1 = self._get_image(obs_file)
        obs2 = self._get_image(obs_file)

        # TODO: Normalize the actions
        action = (action - self.action_mean) / self.action_std

        return obs1, obs2, torch.FloatTensor(action)

    def getitem(self, index): 
        return self.__getitem__(index) # This is to make this method public so that it can be used in animation class 

def get_the_action_means(roots, sec_interval, action_dim):

    train_dset = DynamicsDataset(roots=roots,
                                 sec_interval=sec_interval)
    train_loader = data.DataLoader(train_dset, batch_size=2, shuffle=True)
    pbar = tqdm(total=len(train_loader))
    action_means = torch.FloatTensor(np.zeros(action_dim))
    action_stds = torch.FloatTensor(np.zeros(action_dim))
    num_episode = 0
    for batch in train_loader:
        batch = next(iter(train_loader))
        obs, obs_next, actions = [b for b in batch]

        action_means += actions.mean(dim=0)
        action_stds += actions.std(dim=0)
        print('action means: {}'.format(actions.mean(dim=0)))
        print('action stds: {}'.format(actions.std(dim=0)))
        pbar.update(1)
        num_episode += 1

        print('action_means: {}\naction_stds: {}'.format(
            action_means / num_episode, action_stds / num_episode
        ))

    action_means /= num_episode 
    action_stds /= num_episode 

    pbar.close()

    print('action_means: {}, action_stds: {}'.format(
        action_means, action_stds
    ))

def test_data_aug(roots, sec_interval, action_dim):
    bs = 16
    epochs = 4
    train_dset = DynamicsDataset(roots=roots,
                                 sec_interval=sec_interval)
    train_loader = data.DataLoader(train_dset, batch_size=bs, shuffle=True)
    
    imgs = np.zeros((bs*epochs, 3,480,480))
    for i in range(epochs):
        batch = next(iter(train_loader))
        obs, _, _ = batch

        obs = obs.cpu().detach().numpy()
        # obs_next = obs_next.cpu().detach().numpy()

        imgs[i*bs:(i+1)*bs,:] = obs[:]

    imgs = torch.FloatTensor(imgs)
    save_image(imgs, 'data_aug_try.png', nrow=bs)
    

if __name__ == '__main__':
    roots = glob.glob('data/*') # 6093 is the total number of frames
    sec_interval = 0.5
    action_dim = 2

    test_data_aug(roots, sec_interval, action_dim)


    
    