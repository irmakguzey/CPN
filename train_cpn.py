import argparse 
import os
from os.path import join, exists

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 

import torch 
import torch.nn.functional as F 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as data 

import models as cm 
from dataset import DynamicsDataset

def get_dataloaders():
    train_dset = DynamicsDataset(root='data/28012018_111425')
    train_loader = data.DataLoader(train_dset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # TODO: Add test data loader 

    return train_loader

def compute_infonce_loss(obs, obs_next, encoder, trans, action, device):
    # TODO: bi de bunu kendin hesaplasana
    bs = obs.shape[0] 

    z, z_next = encoder(obs), encoder(obs_next) # b x z_dim 
    z_next_predict = trans(z, action)  # b x z_dim

    neg_dot_products = torch.mm(z_next_predict, z.t()) # b x b
    neg_dists = -((z_next_predict ** 2).sum(1).unsqueeze(1) - 2*neg_dot_products + (z ** 2).sum(1).unsqueeze(0))

    idxs = np.arange(bs)
    neg_dists[idxs, idxs] = float('-inf') # b x b+1

    pos_dot_products = (z_next * z_next_predict).sum(1) # b
    pos_dists = -((z_next**2).sum(1) - 2*pos_dot_products + (z_next_predict ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1) # b x 1 

    dists = torch.cat((neg_dists, pos_dists), dim=1)
    dists = F.log_softmax(dists, dim=1)
    loss = -dists[:,-1].mean() # NOTE: expected yapan sey burda bu

    return loss

def train(train_loader, encoder, trans, optimizer, epoch, device, batch_size=64): # TODO: add train_loader
    encoder.train()
    trans.train() 

    pbar = tqdm(total=len(train_loader.dataset)) # NOTE: For now batch size will be 1
    parameters = list(encoder.parameters()) + list(trans.parameters())

    train_losses = []
    for batch in train_loader:
        obs, obs_next, actions = [el.to(device) for el in batch]
        # print(f'got the batch: obs.shape: {obs.shape}, obs_next.shape: {obs_next.shape}, actions.shape: {actions.shape}')

        # Calculate loss
        loss = compute_infonce_loss(obs, obs_next, encoder, trans, actions, device) 

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step() 

        avg_loss = np.mean(loss.item())
        train_losses.append(avg_loss)

        pbar.set_description(f'Epoch {epoch}, Train loss: {avg_loss:.4f}')
        pbar.update(obs.shape[0])

    pbar.close()
    return train_losses

def main():
    # Load the images 
    # TODO: This will be done with dataloaders later
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    print('set seeds')

    # out_folder_name = join(args.root, 'out') # The loss and similarity plots will be dumped here
    # if not exists(out_folder_name):
    #     os.mkdir(out_folder_name)

    # TODO: get the dataloaders
    train_loader = get_dataloaders()

    print('got dataloaders')

    obs_dim = (3, 480, 640)
    action_dim = 2

    device = torch.device('cuda') # TODO: change this
    print('device: {}'.format(device))
    print('torch.cuda.get_device_name(): {}'.format(torch.cuda.get_device_name()))
    x = torch.FloatTensor([4,5,6]).to(device)
    # print('x: {}'.format(x))

    encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
    trans = cm.Transition(args.z_dim, action_dim).to(device)
    parameters = list(encoder.parameters()) + list(trans.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    print('models initialized')

    train_losses = []
    for epoch in range(args.epochs):
        # Train
        print('epoch in main: {}'.format(epoch))
        train_loss = train(train_loader, encoder, trans, optimizer, epoch, device)
        train_losses += train_loss # All train losses for each episode will be plotted

    
    plt.plot(range(len(train_losses)), train_losses)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters 
    parser.add_argument('--root', type=str, default='data/28012018_111425')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=10)

    # InfoNCE Parameters
    # Negative Samples = Batch Size
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=8) # NOTE: Is 4 enough? 

    args = parser.parse_args() 

    main()