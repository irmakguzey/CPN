# Simclr implementation to train the encoders - if the current model doesn't work
import cv2
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms 

from torchvision.models import resnet18 # NOTE: In actual code resnet50_bn is used
from tqdm import tqdm 

# Forward model that is used and will be thrown away afterwards
class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        # With this forward model zdim will be output_dim!!
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Flatten(), # NOTE: This wasn't created this way in the actual code
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            # nn.SyncBatchNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)



class SimCLR(nn.Module):
    def __init__(self, encoder, projection, loss_temperature, device):
        super(SimCLR, self).__init__()

        # self.encoder = models.resnet18(pretrained=True) 
        self.encoder = encoder
        # n_features = self.encoder.fc.in_features
        # self.projection = Projection(input_dim=n_features)
        self.projection = projection
        self.loss_temperature = loss_temperature
        self.device = device

    def nt_xent_loss(self, out_1, out_2, temperature):
        # out_1, out_2: two different batches with two augmented observations
        # out_1 yani zi, out_2 zj oluyor

        # Bunlarin cat olmus hallerini sonra carpip similarity matrix cikartiyoruz
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix 
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temperature)

        # Remove the sim(zi,zi) from the negative samples
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Get the positive samples
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0) 

        loss = -torch.log( pos / neg ).mean()
        return loss 

    def shared_step(self, batch):
        # We will get two imgs from the simclr dataset
        obs1, obs2, action = [el.to(self.device) for el in batch] # This batch will be taken from the simclr dataset

        # print('obs1.shape: {}, obs2.shape: {}, action.shape: {}'.format(
        #     obs1.shape, obs2.shape, action.shape
        # ))

        h1 = self.encoder(obs1)
        h2 = self.encoder(obs2)

        # print('h1.shape: {}, h2.shape: {}'.format(h1.shape, h2.shape))

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # Project the features to embeddings
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        print('z1.shape: {}, z2.shape: {}'.format(z1.shape, z2.shape))

        loss = self.nt_xent_loss(z1,z2,self.loss_temperature)

        print('loss: {}'.format(loss))

        return loss

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path, _use_new_zipfile_serialization=False)

    # Method to train one epoch
    def train(self, epoch, train_loader, optimizer, rank=0):
        print(f'rank: {rank} in simclr.train')

        self.encoder.train()
        self.projection.train()

        if rank == 0:
            pbar = tqdm(total=len(train_loader)) # NOTE: For now batch size will be 1
        # parameters = list(self.encoder.parameters()) + list(self.projection.parameters())

        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                print(f'rank: {rank} before the self.shared_step, batch[0].shape: {batch[0].shape}')
                loss = self.shared_step(batch)

                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 20)
                optimizer.step()

            train_losses.append(loss.item())
            avg_loss = np.mean(train_losses[-50:])

            # Get the mean of the parameters to see the change
            # mean_params = parameters[0].mean() # Mean of the encoder parameters

            if rank == 0:
                pbar.set_description(f'Epoch {epoch}, Train loss: {avg_loss:.10f}')
                pbar.update(1) # Update for each batch

        if rank == 0: 
            pbar.close()

        return train_losses

    def test(epoch, test_loader, rank=0):
        self.encoder.eval()
        self.projection.eval()

        pbar = tqdm(total=len(test_loader))

        test_loss = []
        for batch in test_loader:
            with torch.no_grad():
                loss = self.shared_step(batch)

                test_loss.append(loss.item())
                avg_loss = np.mean(test_loss[-50:])

            pbar.set_description(f'Test loss: {avg_loss:.10f}, Rank: {rank}')
            pbar.update(1)

        print(f'Epoch {epoch}, Test Loss: {np.mean(test_loss):.4f}')

        return test_loss

        



    