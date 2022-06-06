# Script that makes the forward planning

import numpy as np
import os
import torch
import torch.utils.data as data

from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import DynamicsDataset

class Planner:
    # Planner will sample example ending and in each step example actions and embeddings
    # then it will find the closest observation to that embedding
    def __init__(self, num_trial, len_traj, num_traj, sec_interval, poly_max_deg,
                 test_roots, device, model_load_file, dump_images, dump_images_file=None):
        # Initialize the embedding size
        self.z_dim = 8 # Dimension of the embeddings
        self.sec_interval = sec_interval
        self.poly_max_deg = poly_max_deg
        self.action_dim = 2*(poly_max_deg+1) # Dimension of the actions
        # self.sec_interval = 2 # Will be used when we're fitting the action
        # self.fps = 20 
        # self.action_mean = np.array([-1.3e-2, 4.0e-2, -5.0e-2, 4.0e-2, 2.0e+1, 5.9e+3, 2.9e-4, -1.1e-3,
        #                              1.2e-3, 5.0e-4, -9.6e-4, -1.2e-2]) # These values are taken by looking at the data
        # self.action_std = np.array([3.9e+0, 1.9e+1, 3.4e+1, 2.5e+1, 7.0e+0, 3.5e+3, 4.2e-2, 2.1e-1, 3.7e-1,
        #                             2.8e-1, 1.0e-1, 1.4e-1])

        self.action_mean = np.array([
            -1.7e-5, 5.0e-4, -6.0e-3, 3.7e-2, -1.4e-1,
            3.5e-1, -4.5e-1, 3.0e-1, -6.3e-2, 2.0e+1,
            5.8e+3, 7.2e-7, -1.8e-5, 2.1e-4, -1.1e-3,
            4.3e-3, -1.0e-2, 1.5e-2, -1.2e-2, 5.4e-3,
            -6.0e-4, -9.8e-3
        ])
        self.action_std = np.array([
            1.2e-2, 3.0e-1, 3.2e+0, 1.9e+1, 6.8e+1, 1.5e+2,
            2.1e+2, 1.7e+2, 7.0e+1, 1.2e+1, 3.5e+3, 1.4e-4,
            3.6e-3, 3.8e-2, 2.3e-1, 8.0e-1, 1.8e+0, 2.4e+0,
            1.9e+0, 7.6e-1, 1.4e-1, 1.5e-1
        ])

        self.num_traj = num_traj # Number of trajectories to plan
        self.len_traj = len_traj # Length of each trajectory
        self.num_trial = num_trial # Number of predicted embeddings in each step before finding the closest one to the desired embedding
        self.dump_images = dump_images
        self.dump_images_file = dump_images_file

        # Initialize the test data loader with the given test roots
        self.bs = 128 
        self.test_dset = DynamicsDataset(roots=test_roots,
                                         sec_interval=sec_interval,
                                         poly_max_deg=poly_max_deg)
        self.test_loader = data.DataLoader(self.test_dset, batch_size=self.bs, shuffle=False,
                                           num_workers=4, pin_memory=True)

        # Load the saved model
        self.device = device 
        checkpoint = torch.load(model_load_file, map_location=device)
        self.encoder = DDP(checkpoint['encoder'].to(device), device_ids=[0])
        self.trans = DDP(checkpoint['trans'].to(device), device_ids=[0])

        # Embeddings and actions to be held for each trajectory
        self.traj_zs = np.zeros((self.num_traj, self.len_traj+1, self.z_dim))
        self.traj_actions = np.zeros((self.num_traj, self.len_traj, self.action_dim)) # There will be one less action then the observations

        # Initialize the inverse transform to be used for observations
        self.inv_trans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                  std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                             transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                                  std = [ 1., 1., 1. ]),
                                            ])

    def get_all_embeddings(self):
        pbar = tqdm(total=len(self.test_loader))
        # Create arrays to get all the embeddings
        self.Z_curr = np.zeros((len(self.test_dset), self.z_dim))
        self.Z_next = np.zeros((len(self.test_dset), self.z_dim))
        # self.Z_next_predict = np.zeros((len(self.test_dset), self.z_dim))
        ep = 0
        for batch in self.test_loader:
            # Get current batch
            obs, obs_next, actions = [el.to(self.device) for el in batch]

            # Get the current embeddings
            z, z_next = self.encoder(obs), self.encoder(obs_next) # NOTE: we don't really need predicted embeddings here
            # z_next_predict = self.trans(z, actions)

            self.Z_curr[ep*self.bs:(ep+1)*self.bs, :] = z[:,:].cpu().detach().numpy()
            self.Z_next[ep*self.bs:(ep+1)*self.bs, :] = z_next[:,:].cpu().detach().numpy()
            # self.Z_next_predict[ep*self.bs:(ep+1)*self.bs, :] = z_next_predict[:,:].cpu().detach().numpy()

            ep += 1
            pbar.update(1)

        pbar.close()
        
    def choose_random_traj(self):
        # Chooses randomly from self.Z_curr and self.Z_next as the start and end goals of the trajectory
        # Get random starting embeddings
        start_rand_idx = np.random.choice(range(self.Z_curr.shape[0]), self.num_traj)
        end_rand_idx = np.random.choice(range(self.Z_curr.shape[0]), self.num_traj)
        self.z_starts = self.Z_curr[start_rand_idx, :]
        self.z_ends = self.Z_curr[end_rand_idx, :]

        print('self.z_starts.shape: {}, self.z_ends.shape: {}'.format(
            self.z_starts.shape, self.z_ends.shape
        ))

    # Sample one action from a gaussian distribution with the given means
    # and stds
    # TODO: This can be changed to do knn imitation
    def sample_action(self):
        rand_act = np.random.normal(self.action_mean, self.action_std)
        # rand_act = np.expand_dims(rand_act, 0) # Shape: (1, action_dim)
        # print('rand_act.shape: {}'.format(rand_act.shape))
        return rand_act 

    # Method to plan one step for one trajectory's embeddings
    # It sample action for each trial, predict the next embedding
    # And find the closest observation to that predicted next embedding
    # z_curr, z_end's shape is (1, z_dim)
    def single_plan(self, z_curr, z_end):
        zs, actions = [], []
        
        with torch.no_grad():
            for _ in range(self.num_trial): 
                action = torch.FloatTensor(self.sample_action()).to(self.device)
                z_next = self.trans(z_curr, action)
                zs.append(z_next)
                actions.append(action)
            
            # Calculate the l2 distances between z_end and zs - NOTE: this might be wrong
            zs = torch.stack(zs, dim=0)
            # print('zs.shape: {}'.format(zs.shape))
            # print('z_end.shape: {}'.format(z_end.shape))
            dists = torch.norm(zs - z_end, dim=1, p=None)
            # print('dists.shape: {}'.format(dists.shape))
            idx = torch.argmin(dists)
            # print('idx: {}'.format(idx))
            return actions[idx], zs[idx]

    # Plan trajectory for one trajectory and update self.traj_actions/zs arrays
    def plan_traj(self, traj_id, z_start, z_end):
        # z_start/z_end.shape: (1, z_dim)
        self.traj_zs[traj_id,0,:] = z_start[:] 
        z_start = torch.FloatTensor(z_start).to(self.device)
        z_end = torch.FloatTensor(z_end).to(self.device)
        for step in range(1, self.len_traj):
            print(f'Step: {step}')
            action, z_start = self.single_plan(z_start, z_end) # Update the z_start in each step
            self.traj_actions[traj_id, step-1, :] = action.cpu().detach().numpy()
            self.traj_zs[traj_id, step, :] = z_start.cpu().detach().numpy()

        self.traj_zs[traj_id, -1, :] = z_end.cpu().detach().numpy() # Add the final embedding to show

    # Plan all the trajectories - and dumps images if required
    def plan_all(self):
        # for each wanted trajectory call plan_traj as the starting embedding from z_starts and z_ends
        for traj_id in range(self.num_traj):
            z_start = self.z_starts[traj_id, :]
            z_end = self.z_ends[traj_id, :]
            self.plan_traj(traj_id, z_start, z_end)
            print(f'----\nTrajectory {traj_id} planned\n----')

        if self.dump_images:
            self.dump_trajectories()

    # finds the closest embeddings for each predicted embedding and then retrieves
    # the corresponding observation
    def dump_trajectories(self):
        imgs = np.zeros((self.num_traj, self.len_traj+1, 3, 480, 480)) # Last one will be the wanted observation - Will be straightened to -1,480,480,3
        for traj_id in range(self.num_traj):
            for step in range(self.len_traj+1): # TODO: 5-10 icin de yap
                # For each embedding in self.traj_zs, find the closest embedding in self.Z_next
                curr_z_next_predict = self.traj_zs[traj_id, step, :]
                dist = np.linalg.norm(self.Z_next - curr_z_next_predict, axis=1)
                # print('dist.shape: {}'.format(dist.shape))
                closest_z_idx = np.argsort(dist)[0]

                # Get the closest observation
                obs, _, _ = self.test_dset.getitem(closest_z_idx)
                obs = self.inv_trans(obs) # Inverse transform the observation

                # print('obs.shape: {}'.format(obs.shape))

                imgs[traj_id, step, :] = obs[:]

        imgs = np.reshape(imgs, (-1, 3,480,480)) # Flatten the first dimension
        imgs = torch.FloatTensor(imgs)
        save_image(imgs, self.dump_images_file, nrow=self.len_traj+1)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # Start the multiprocessing to load the saved models properly
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    torch.cuda.set_device(0)

    test_roots = [
        "data/28012018_111041",
        "data/28012018_124452",
        "data/29032022_195715",
        "data/28012018_122358",
        "data/28012018_120304",
        "data/28012018_124425"
    ]

    planner = Planner(
        num_trial=100, # For each step it should sample and guess 100 different actions
        len_traj=20, # It should try to reach in 10 steps
        num_traj=10,
        sec_interval=5,
        poly_max_deg=10,
        test_roots=test_roots,
        device=torch.device('cuda:0'),
        model_load_file='out/train_sec_5_adim_22_zdim_8/checkpoint_40.pt',
        dump_images=True,
        dump_images_file='train_sec_5_adim_22_zdim_8_planning.png'
    )

    planner.get_all_embeddings()
    planner.choose_random_traj()
    planner.plan_all()

