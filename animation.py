import cv2
import os
from os.path import join, dirname, basename
import matplotlib
import math
import glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch 
# import torch.nn.functional as F 
# import torch.nn as nn 
# import torch.optim as optim 
import torch.utils.data as data 

from matplotlib.animation import FuncAnimation, FFMpegWriter
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import DynamicsDataset

# Animation wrapper for plotting actions

class AnimationWrapper:
    def __init__(self, X, Y_pred, Y_act, dump_dir, is_image=False, dump_file='animation.mp4', total_frames=2000, sec_interval=1, fps=20): # directory to dump the animated video
        # if Y is none then poly method will be used to construct Y, else Y will be plotted

        self.X = X # X can be none when there is an image to be given
        self.Y_pred = Y_pred # Y_pred can have multiple images if there are multiple nearest neighbours given
        self.Y_act = Y_act

        self.sec_interval = sec_interval
        self.frame_interval = fps * sec_interval
        self.is_image = is_image # boolean to indicate if the data we're animating is an image or not

        k = Y_pred.shape[1]
        nrows = 2
        # ncols = math.ceil((k+2) / nrows)
        ncols = 3 # NOTE: Simdilik k'yi biliyomus gibi davranicam :D 
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=nrows, ncols=ncols) 

        if is_image:
            nrows = 2
            # ncols = math.ceil((k+2) / nrows)
            ncols = 3 # NOTE: Simdilik k'yi biliyomus gibi davranicam :D 
            self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=nrows, ncols=ncols) 

            self.imgs = []
            self.imgs.append([self.axs[0,0].imshow(X[0], cmap='gray')])
            self.imgs[0].append(self.axs[0,1].imshow(Y_act[0], cmap='gray'))
            self.imgs[0].append(self.axs[0,2].imshow(Y_pred[0,0], cmap='gray'))

            self.imgs.append([self.axs[1,0].imshow(Y_pred[0,1], cmap='gray')])
            self.imgs[1].append(self.axs[1,1].imshow(Y_pred[0,2], cmap='gray'))
            self.imgs[1].append(self.axs[1,2].imshow(Y_pred[0,3], cmap='gray'))

            self.axs[0,0].set_title("Current Observation")
            self.axs[0,1].set_title("Next Observation")
            self.axs[0,2].set_title("1st NN")

            self.axs[1,0].set_title("2nd NN")
            self.axs[1,1].set_title("3rd NN")
            self.axs[1,2].set_title("4th NN")

        else:
            self.fig, self.axs = plt.subplots(nrows=1,ncols=2)

            self.line_actual, = self.axs[0].plot([], [])
            self.line_pred, = self.axs[1].plot([], [])
            self.axs[0].set_ylim(-0.5,0.5)
            self.axs[0].set_title("Actual Action")

            self.axs[1].set_ylim(-0.5,0.5)
            self.axs[1].set_title("Predicted Action")

        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = total_frames
        )

        self.anim.save(join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('animation saved to: {}'.format(join(dump_dir, dump_file)))

    def init_fun(self):
        if self.is_image:
            # self.img_actual.set_array(np.zeros((self.Y_act[0].shape)))
            # self.img_pred.set_array(np.zeros((self.Y_pred[0].shape)))

            # self.imgs.append([self.axs[0,0].imshow(X[0], cmap='gray')])
            # self.imgs[0].append(self.axs[0,1].imshow(Y_act[0], cmap='gray'))
            # self.imgs[0].append(self.axs[0,2].imshow(Y_pred[0,0], cmap='gray'))

            # self.imgs.append([self.axs[1,0].imshow(Y_pred[0,1], cmap='gray')])
            # self.imgs[1].append(self.axs[1,1].imshow(Y_pred[0,2], cmap='gray'))
            # self.imgs[1].append(self.axs[1,2].imshow(Y_pred[0,3], cmap='gray'))

            self.imgs[0][0].set_array(np.zeros((self.X[0].shape)))
            self.imgs[0][1].set_array(np.zeros((self.Y_act[0].shape)))
            self.imgs[0][2].set_array(np.zeros((self.Y_pred[0,0].shape)))

            self.imgs[1][0].set_array(np.zeros((self.Y_pred[0,1].shape)))
            self.imgs[1][1].set_array(np.zeros((self.Y_pred[0,2].shape)))
            self.imgs[1][2].set_array(np.zeros((self.Y_pred[0,3].shape)))

            # return self.img_actual, self.img_pred,
            self.imgs[0][0], self.imgs[0][1], self.imgs[0][2], self.imgs[1][0], self.imgs[1][1], self.imgs[1][2],
        else:
            self.line_actual.set_data([], [])
            self.line_pred.set_data([], [])
            return self.line_actual, self.line_pred,

    def animate(self, i):

        if self.is_image:
            x = self.X[i]
            y_pred = self.Y_pred[i]
            y_act = self.Y_act[i]

            self.imgs[0][0].set_array(x)
            self.imgs[0][1].set_array(y_act)
            self.imgs[0][2].set_array(y_pred[0])

            self.imgs[1][0].set_array(y_pred[1])
            self.imgs[1][1].set_array(y_pred[2])
            self.imgs[1][2].set_array(y_pred[3])

            # self.img_actual.set_array((y_act * 255).astype(np.uint8))
            # self.img_pred.set_array((y_pred * 255).astype(np.uint8))

            return self.imgs[0][0], self.imgs[0][1], self.imgs[0][2], self.imgs[1][0], self.imgs[1][1], self.imgs[1][2],

        else:
            x = self.X[i:i+self.frame_interval]
            y_pred = self.Y_pred[i]
            y_act = self.Y_act[i]

            # print('x: {}, y_pred: {}, y_act: {}'.format(
            #     x.shape, y_pred.shape, y_act.shape
            # ))
            
            self.axs[0].set_xlim(min(x), max(x))
            self.axs[1].set_xlim(min(x), max(x))
            # self.axis.set_ylim(min(y), max(y))

            self.line_actual.set_data(x, y_act)
            self.line_pred.set_data(x, y_pred)

            return self.line_actual, self.line_pred, 

# Method to load a trained model (model_load_file) and visualize its outputs with the given fps 
def visualize_test(test_roots, model_load_file, dump_file, device, k,
                   total_frames, sec_interval, poly_max_deg, fps=1): # In order to differentiate the predicted and actual observation fps should be low
    test_dset = DynamicsDataset(roots=test_roots,
                                sec_interval=sec_interval,
                                poly_max_deg=poly_max_deg)
    bs = 128
    test_loader = data.DataLoader(test_dset, batch_size=bs, # TODO: Change this to: len(test_dset)
                                  shuffle=False, num_workers=4, pin_memory=True)
    
    # Load the saved model
    checkpoint = torch.load(model_load_file, map_location=device)
    encoder = DDP(checkpoint['encoder'].to(device), device_ids=[0])
    trans = DDP(checkpoint['trans'].to(device), device_ids=[0])

    # Embeddings will be stacked for all the observations in the test set
    pbar = tqdm(total=len(test_loader))
    Z_next = np.zeros((len(test_dset), 8)) 
    Z_next_predict = np.zeros((len(test_dset), 8))
    ep = 0
    for batch in test_loader:
        # Get current batch
        obs, obs_next, actions = [el.to(device) for el in batch]

        # Get the current embeddings
        z, z_next = encoder(obs), encoder(obs_next)
        z_next_predict = trans(z, actions)

        Z_next[ep*bs:(ep+1)*bs, :] = z_next[:,:].cpu().detach().numpy()
        Z_next_predict[ep*bs:(ep+1)*bs, :] = z_next_predict[:,:].cpu().detach().numpy()

        pbar.update(1)
        ep += 1
        pbar.set_description('ep*bs: {}, (ep+1)*bs: {}'.format(ep*bs, (ep+1)*bs))

    pbar.close()

    # Inverse transform for obs and obs_next 
    inv_trans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                         std = [ 1., 1., 1. ]),
                                  ])

    X = np.zeros((total_frames, 480, 480, 3))
    Y_act = np.zeros((total_frames, 480, 480, 3))
    Y_pred = np.zeros((total_frames, k, 480, 480, 3))

    rand_indices = np.random.choice(range(len(Z_next_predict)), (total_frames,))
    for i,z_index in enumerate(rand_indices): # Traverse through these embeddings and compare them to the whole dataset
        curr_z_next_predict = Z_next_predict[z_index:z_index+1, :] # SHape: (1, z_dim)
        dist = np.linalg.norm(Z_next - curr_z_next_predict, axis=1)
        nearest_z_indices = np.argsort(dist)[:k] # Get the first k indices

        # Stack to X and Y_act for ith obs and obs_next
        obs, obs_next, _ = test_dset.getitem(z_index)
        obs, obs_next = inv_trans(obs), inv_trans(obs_next)

        X[i,:,:,:] = obs.permute(1,2,0).cpu()
        Y_act[i,:,:,:] = obs_next.permute(1,2,0).cpu()

        for j, kth_z_index in enumerate(nearest_z_indices):
            _, obs_next, _ = test_dset.getitem(kth_z_index)
            obs_next = inv_trans(obs_next)
            Y_pred[i,j,:] = obs_next.permute(1,2,0).cpu()

    AnimationWrapper(
        X = X,
        Y_pred = Y_pred,
        Y_act = Y_act,
        is_image=True,
        dump_dir = 'animations',
        dump_file = dump_file,
        total_frames = total_frames,
        fps=fps, # which should be quite low
    )


if __name__ == '__main__':
    # total_frames = 200
    # P = np.random.rand(total_frames, 3)
    # X = np.linspace(0, 10, total_frames)

    # if not os.path.isdir('animations'):
    #     os.mkdir('animations')
    # anim_wrap = AnimationWrapper(
    #     X = X,
    #     Y = np.sin(X),
    #     dump_dir='animations',
    #     total_frames = total_frames
    # )

    # Trying the animation with images
    # images_folder = 'data/28012018_113050/images'
    # image_names = glob.glob(join(images_folder, 'frame*'))
    # image_names = sorted(image_names)

    # # Get random 200 images for both Y_pred and Y_act 
    # total_frames = 200
    # # pred_img_names = np.random.choice(image_names, total_frames)
    # # act_img_names = np.random.choice(image_names, total_frames)
    # pred_img_names = image_names[:200]
    # act_img_names = image_names[-200:]

    # y_pred = cv2.imread(pred_img_names[0])
    # y_pred = cv2.cvtColor(src=y_pred, code=cv2.COLOR_BGR2GRAY)
    # print(y_pred.shape)

    # Y_pred = np.zeros((total_frames, y_pred.shape[0], y_pred.shape[1]))
    # Y_act = np.zeros((total_frames, y_pred.shape[0], y_pred.shape[1]))
    # print('Y_pred.shape: {}'.format(Y_pred.shape))

    # for i in range(total_frames):
    #     pred_img = cv2.imread(pred_img_names[i])
    #     pred_img = cv2.cvtColor(src=pred_img, code=cv2.COLOR_BGR2GRAY)
    #     Y_pred[i,:,:] = pred_img[:,:]

    #     act_img = cv2.imread(act_img_names[i])
    #     act_img = cv2.cvtColor(src=act_img, code=cv2.COLOR_BGR2GRAY)
    #     Y_act[i,:,:] = act_img[:,:]

    # anim_wrap = AnimationWrapper(
    #     Y_pred = Y_pred, 
    #     Y_act = Y_act,
    #     dump_dir = 'animations',
    #     dump_file = '28012018_113050_rand_imgs.mp4',
    #     total_frames = total_frames,
    #     is_image = True,
    # )

    torch.cuda.empty_cache()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    torch.cuda.set_device(0)

    visualize_test(
        test_roots = [
            "data/24052022_193947",
            "data/28012018_125252"
        ],
        poly_max_deg=0,
        sec_interval=3,
        model_load_file = 'out/train_25052022_225735/checkpoint_50.pt',
        dump_file='train_sec_3_adim_2_man.mp4',
        device = torch.device('cuda:0'), # it was trained on cuda:1 before 
        k=4, # predictions up to 4th nearest neighbour will be shown 
        total_frames=128,
    )



