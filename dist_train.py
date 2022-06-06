# Script to do the training distributed across all GPUs using DDP package of Pytorch. 
import argparse
import glob
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as data 

from datetime import datetime
from os.path import join, exists
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

# My own libraries
import models as cm 
from dataset import DynamicsDataset
from train_cpn import test, train

def create_roots(train_dir):
    roots = glob.glob('data/*') # Get the roots to get data from - TODO: change this - get all of the roots
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

    print('test_roots: {}'.format(test_roots))
    print('train_roots: {}'.format(train_roots))

    if not os.path.isfile(join(train_dir, 'test_roots.json')): # test roots will be the same for all the roots (when all the data is used)
        # Serializing json 
        test_roots_json = {
            'test_roots': test_roots
        }
        json_object = json.dumps(test_roots_json, indent = 4)
        
        # Writing to sample.json
        with open(join(train_dir, 'test_roots.json'), "w") as outfile:
            outfile.write(json_object)

    return train_roots, test_roots

def get_dataloaders(train_roots, test_roots, args): # world_size: Number of possible GPUs 

    test_dset = DynamicsDataset(roots=test_roots,
                                sec_interval=args.sec_interval,
                                poly_max_deg=args.poly_max_deg)
    test_sampler = data.DistributedSampler(test_dset)
    test_loader = data.DataLoader(test_dset,
                                  batch_size=args.batch_size,
                                  shuffle=False, # This should be false when DistributedSampler is used, here if true it will shuffle the subsamples
                                  num_workers=4,
                                  sampler=test_sampler)

    train_dset = DynamicsDataset(roots=train_roots,
                                 sec_interval=args.sec_interval,
                                 poly_max_deg=args.poly_max_deg)
    train_sampler = data.DistributedSampler(train_dset)
    train_loader = data.DataLoader(train_dset,
                                  batch_size=args.batch_size,
                                  shuffle=False, # This should be false when DistributedSampler is used, here if true it will shuffle the subsamples
                                  num_workers=4,
                                  sampler=train_sampler)

    return train_loader, test_loader

# General method to start multi GPUd training
def start_process(rank, world_size, args, train_dir, train_roots, test_roots):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.barrier() # Wait for all of the processes to start

    # Get dataloader for this rank
    print('rank: {}'.format(rank))

    if rank == 0:
        writer = SummaryWriter(train_dir)

    # print('len(train_loader): {}'.format(len(train_loader))) # This will be divided by the number of GPUs used - originally number of batches is around 1900, with multi processing it's lowered to 500 for each GPU
    train_loader, test_loader = get_dataloaders(train_roots, test_roots, args)
    print('len(train_loader): {}, len(train_loader.dataset): {}'.format(len(train_loader), len(train_loader.dataset)))
    # Create the models with the given rank and device
    obs_dim = (3, 480, 480)
    # action_dim = 2*(args.poly_max_deg+1)
    action_dim = 2

    # Create CUDA device with the given rank
    device = torch.device(f'cuda:{rank}')

    # Initialize models
    encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
    trans = cm.Transition(args.z_dim, action_dim).to(device)
    print('encoder: {}\ntrans: {}'.format(encoder, trans))
    parameters = list(encoder.parameters()) + list(trans.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    # Wrap the models in DDP wrapper
    encoder = DDP(encoder, device_ids=[rank], output_device=rank)
    trans = DDP(trans, device_ids=[rank], output_device=rank)

    # Train losses and model checkpoints are only saved when the rank == 0 since the model parameters
    # and gradients are distributed between GPUs and copy of the model is always preserved in all GPUs
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        # Train
        print('rank: {} entered -before training- barier'.format(rank))      
        dist.barrier() # All GPUs should start training at the same time  
        train_loss = train(train_loader, encoder, trans, optimizer, epoch, device, args.batch_size, rank)

        print('rank: {} entered -after training- barier'.format(rank))
        dist.barrier() # Wait until all the processes reached here
    
        if rank == 0: # Add the train loss to the tensorboard writer
            for i in range(len(train_loss)-1):
                writer.add_scalar('Train Loss', train_loss[i], epoch * args.batch_size + i)

        if rank == 0 and epoch % args.test_interval == 0:
            print('rank: {}, testing epoch: {}'.format(rank, epoch))
            test_loss, sim = test(test_loader, encoder, trans, epoch, device, rank)
            for i in range(len(test_loss)-1):
                writer.add_scalar('Test Loss', test_loss[i], (epoch / args.test_interval) * args.batch_size + i)
                writer.add_scalar('Euclidean Dist', sim[0], (epoch / args.test_interval) * args.batch_size + i)
                writer.add_scalar('Cosine Similarity', sim[1], (epoch / args.test_interval) * args.batch_size + i)
                writer.add_scalar('Dot Product', sim[2], (epoch / args.test_interval) * args.batch_size + i)

        if rank == 0 and epoch % args.model_save_interval == 0:
            checkpoint = {
                'encoder': encoder,
                'trans': trans,
                'optimizer': optimizer,
            }
            torch.save(checkpoint, join(train_dir, f'checkpoint_{epoch}.pt'), _use_new_zipfile_serialization=False)
            # Save models and optimizers' state_dict so that it would be easier to load in the car 
            torch.save(encoder.state_dict(), join(train_dir, f'encoder_{epoch}.pt'), _use_new_zipfile_serialization=False)
            torch.save(trans.state_dict(), join(train_dir, f'trans_{epoch}.pt'), _use_new_zipfile_serialization=False)
            torch.save(optimizer.state_dict(), join(train_dir, f'optimizer_{epoch}.pt'), _use_new_zipfile_serialization=False)

    if rank == 0:
        writer.close()


def main():
    # Create the directory to save the outputs
    if args.train_dir == 'train': # We should create a train_dir with timestamp if there is no additional value given
        now = datetime.now()
        time_str = now.strftime('%d%m%Y_%H%M%S')
        train_dir = join(args.train_out, 'train_{}'.format(time_str))
        
    else:
        train_dir = join(args.train_out, args.train_dir)
    os.mkdir(train_dir)
    train_roots, test_roots = create_roots(train_dir)
    
    n_gpus = torch.cuda.device_count()
    print('args: {}'.format(args))
    print('n_gpus = world_size: {}'.format(n_gpus))
    world_size = 4
    mp.spawn(start_process,
        args=(world_size, args, train_dir, train_roots, test_roots),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Dataset Parameters 
    parser.add_argument('--root', type=str, default='data/28012018_111425')
    parser.add_argument('--train_out', type=str, default='out')
    parser.add_argument('--train_dir', type=str, default='train')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=51)
    parser.add_argument('--test_interval', type=int, default=2)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--model_save_interval', type=int, default=5)
    parser.add_argument('--model_load', type=bool, default=False)
    parser.add_argument('--model_load_file', type=str, default='checkpoint_100.pt')

    # InfoNCE Parameters
    # Negative Samples = Batch Size
    parser.add_argument('--batch_size', type=int, default=256) # TODO: Change this
    # parser.add_argument('--action_dim', type=int, default=12)
    parser.add_argument('--poly_max_deg', type=int, default=10)
    parser.add_argument('--sec_interval', type=int, default=0.5)
    parser.add_argument('--z_dim', type=int, default=64) 
    parser.add_argument('--name', type=str, default='arya')
    parser.add_argument('--seed', type=int, default=17)

    args = parser.parse_args() 

    main()