import argparse 
import glob
import os
from os.path import join, exists
from datetime import datetime

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
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
    roots = glob.glob('data/*') # 6093 is the total number of frames
    test_root_num = math.floor(len(roots) * args.test_ratio)

    # Find random indices to get from roots for test
    test_rand_indices = np.random.choice(np.arange(len(roots)), test_root_num)
    test_roots = []
    train_roots = []
    for i, root in enumerate(roots):
        if i in test_rand_indices:
            test_roots.append(root)
        else: 
            train_roots.append(root)

    test_dset = DynamicsDataset(roots=test_roots,
                                sec_interval=args.sec_interval,
                                poly_max_deg=args.poly_max_deg)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

    train_dset = DynamicsDataset(roots=train_roots,
                                 sec_interval=args.sec_interval,
                                 poly_max_deg=args.poly_max_deg)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def compute_infonce_loss(obs, obs_next, encoder, trans, action):
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

def get_l2_norm(z, dim=1):
    return z.pow(2).sum(dim=dim).sqrt()

# def compute_infonce_loss(obs, obs_next, encoder, trans, actions):
#     batch_size = obs.shape[0] 

#     z, z_pos = encoder(obs), encoder(obs_next) # Shape: (B, Z_dim)
#     # print('z.shape: {}'.format(z.shape))
#     z_next = trans(z, actions) # Shape: (B, Z_dim)
#     # print('z_next.shape {}'.format(z_next.shape)) 

#     # For now let's use z as the negative samples
#     # z_neg = torch.cat((z, z_pos), dim=1) # Shape: (B, 2*Z_dimThere will be two negative samples for each embedding
#     z_neg = z.unsqueeze(0).repeat(batch_size, 1, 1) # Shape: (B,B,Z_dim) - For each positive embedding we will have B number of negative embeddings
#     # print('z_neg.shape: {}'.format(z_neg.shape))

#     h_pos = torch.exp(- get_l2_norm(z_next - z_pos) ** 2) # Shape: (B) - since it is summed
#     # print('h_pos.shape: {}'.format(h_pos.shape))

#     h_neg = torch.exp(- get_l2_norm((z_next - z_neg), dim=2) ** 2) # It should be summed in the second dimension
#     # print('h_neg.shape: {}'.format(h_neg.shape))
    
#     h_neg_sum = h_neg.sum(dim=1)
#     # print('h_neg_sum: {}'.format(h_neg_sum))
#     # print('h_neg_sum.shape: {}'.format(h_neg_sum.shape))

#     # print(torch.div(h_pos, h_neg_sum).shape)
#     log_out = torch.log(torch.div(h_pos, h_neg_sum)) NOTE: Buradaki division nan'a sebep oluyor! 
#     # print('log_out: {}'.format(log_out))
#     # print('log_out.shape: {}'.format(log_out.shape))

#     loss = - log_out.mean()
#     # print('loss: {}'.format(loss))

#     return loss

# These functions are used to calculate similarity between two embeddings
def euclidean_dist(z1, z2):
    norm_z1 = F.normalize(z1)
    norm_z2 = F.normalize(z2)

    z1_reshaped = norm_z1.reshape(-1)
    z2_reshaped = norm_z2.reshape(-1)

    return ((z1_reshaped - z2_reshaped) ** 2).sum().sqrt()

def cosine(z1, z2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    norm_z1 = F.normalize(z1)
    norm_z2 = F.normalize(z2)

    return cos(norm_z1,norm_z2).sum()

def dot_product(z1, z2):
    norm_z1 = F.normalize(z1)
    norm_z2 = F.normalize(z2)

    mm = torch.mm(norm_z1, norm_z2.t())
    diag = torch.diagonal(mm) # Will give dot product
    return diag.sum() 

def test(test_loader, encoder, trans, epoch, device, rank=0):
    encoder.eval()
    trans.eval()

    pbar = tqdm(total=len(test_loader))

    test_loss = []
    euc_dists = 0
    cosines = 0
    dot_products = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_next, actions = [el.to(device) for el in batch]
            # Calculate loss
            loss = compute_infonce_loss(obs, obs_next, encoder, trans, actions)

            # Calculate the similarities
            z, z_next = encoder(obs), encoder(obs_next)
            z_next_predict = trans(z, actions)

            test_loss.append(loss.item())
            avg_loss = np.mean(test_loss[-50:])
            # avg_test_loss = np.mean(test_loss[-50:])
            euc_dists += euclidean_dist(z_next, z_next_predict) * obs.shape[0]
            cosines += cosine(z_next, z_next_predict) * obs.shape[0]
            dot_products += dot_product(z_next, z_next_predict) * obs.shape[0]

        pbar.set_description(f'Test loss: {avg_loss:.10f}, Rank: {rank}')
        pbar.update(1)

    euc_dists /= len(test_loader.dataset)
    cosines /= len(test_loader.dataset)
    dot_products /= len(test_loader.dataset)

    print(f'Epoch {epoch}, Test Loss: {np.mean(test_loss):.4f}, Similarities: (Euclidean Dists: {euc_dists:.4f}, Cosines: {cosines:.4f}, Dot Products: {dot_products:.4f}')

    return test_loss, [euc_dists, cosines, dot_products]


def train(train_loader, encoder, trans, optimizer, epoch, device, batch_size=64, rank=0): # TODO: add train_loader
    encoder.train()
    trans.train() 

    if rank == 0:
        pbar = tqdm(total=len(train_loader)) # NOTE: For now batch size will be 1
    parameters = list(encoder.parameters()) + list(trans.parameters())

    train_losses = []
    for batch in train_loader:
        optimizer.zero_grad() # Set the gradient to zero 
        obs, obs_next, actions = [el.to(device) for el in batch]

        # Calculate loss
        loss = compute_infonce_loss(obs, obs_next, encoder, trans, actions)

        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step() 

        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])

        # Get the mean of the parameters to see the change
        mean_params = parameters[0].mean() # Mean of the encoder parameters

        if rank == 0:
            pbar.set_description(f'Epoch {epoch}, Train loss: {avg_loss:.10f}, Param Mean: {mean_params}, Rank: {rank}')
            pbar.update(1) # Update for each batch

    if rank == 0: 
        pbar.close()
    return train_losses

def main():
    # Load the images 
    # TODO: This will be done with dataloaders later
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_dataloaders()
    print('len(train_loader): {}, len(train_loader.dataset): {}'.format(len(train_loader), len(train_loader.dataset)))

    now = datetime.now()
    time_str = now.strftime('%d%m%Y_%H%M%S')
    train_dir = join(args.train_out, 'train_{}'.format(time_str))
    print('train_dir: {}'.format(train_dir))
    os.mkdir(train_dir)

    obs_dim = (3, 480, 480)
    action_dim = 2*(args.poly_max_deg+1)

    device = torch.device('cuda:1') # TODO: change this
    print('device: {}'.format(device))
    print('torch.cuda.get_device_name(): {}'.format(torch.cuda.get_device_name()))

    if args.model_load:
        checkpoint = torch.load(args.model_load_file)
        encoder = checkpoint['encoder'].to(device)
        trans = checkpoint['trans'].to(device)
        optimizer = checkpoint['optimizer']

    else:
        encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
        trans = cm.Transition(args.z_dim, action_dim).to(device)
        parameters = list(encoder.parameters()) + list(trans.parameters())
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    print('models initialized')

    train_losses = []
    test_losses = []
    similarities = []
    for epoch in range(args.epochs):
        # Train
        print('epoch in main: {}'.format(epoch))
        # print('parameters.shape: {}'.format(parameters[0]))
        train_loss = train(train_loader, encoder, trans, optimizer, epoch, device, args.batch_size)
        train_losses += train_loss[:-1] # The last episode has lower training loss because of the data 
    
        if epoch % args.test_interval == 0:
            test_loss, sim = test(test_loader, encoder, trans, epoch, device)
            test_losses += (test_loss[:-1])
            similarities.append(sim) # Shape: (# of tests, 3)

        if epoch % args.model_save_interval == 0:
            checkpoint = {
                'encoder': encoder,
                'trans': trans,
                'optimizer': optimizer,
            }
            torch.save(checkpoint, join(train_dir, f'checkpoint_{epoch}.pt'))

    plt.plot(range(len(train_losses)), train_losses)
    plt.savefig(join(train_dir, 'train_loss.png'))
    plt.clf()

    plt.plot(range(len(test_losses)), test_losses)
    plt.savefig(join(train_dir, 'test_loss.png'))
    plt.clf()

    sim_np = np.zeros((len(similarities), 2))
    euc_dists = np.zeros((len(similarities), 1))
    for i,sim in enumerate(similarities):
        euc_dists[i,0] = similarities[i][0].cpu()
        sim_np[i,0] = similarities[i][1].cpu()
        sim_np[i,1] = similarities[i][2].cpu()
    plt.plot(range(len(euc_dists)), euc_dists[:,0], label='Euclidean Dists')
    plt.legend()
    plt.savefig(join(train_dir, 'euc_dists.png'))
    plt.clf()

    plt.plot(range(len(sim_np)), sim_np[:,0], label='Cosines')
    plt.legend()
    plt.savefig(join(train_dir, 'cosines.png'))
    plt.clf()

    plt.plot(range(len(sim_np)), sim_np[:,1], label='Dot Products')
    plt.legend()
    plt.savefig(join(train_dir, 'dot_products.png'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters 
    parser.add_argument('--root', type=str, default='data/28012018_111425')
    parser.add_argument('--train_out', type=str, default='out')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--model_save_interval', type=int, default=10)
    parser.add_argument('--model_load', type=bool, default=False)
    parser.add_argument('--model_load_file', type=str, default='checkpoint_100.pt')

    # InfoNCE Parameters
    # Negative Samples = Batch Size
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--poly_max_deg', type=int, default=10)
    parser.add_argument('--sec_interval', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=8) 
    parser.add_argument('--name', type=str, default='arya')
    parser.add_argument('--seed', type=int, default=17)

    args = parser.parse_args() 

    main()