import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

# Class to learn current embeddings
class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()

        print('inside Encoders init')
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            # PrintSize(),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            # PrintSize(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Option 1: 256 x 8 x 8
            # PrintSize(),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
            # PrintSize(),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize()
        )

        self.out = nn.Linear(256 * 3 * 5, z_dim)

        print('done with Encoder')

    def forward(self, x):
        x = self.model(x) # This works perfectly fine
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x

# Model to predict next embeddings given an embedding and an action
class Transition(nn.Module):
    def __init__(self, z_dim, action_dim):
        super().__init__()

        print('inside Transition init')

        self.z_dim = z_dim 
        hidden_dim = 64
        self.model = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
        print('done with Transition')

    def forward(self, z, a):
        x = torch.cat((z,a), dim=-1)
        x = self.model(x)
        return x

# Will be used to decode the predicted next embeddings to actual images
class Decoder(nn.Module):
    def __init__(self, z_dim, channel_dim): # NOTE: don't think the other args are needed - check if this is true
        super().__init__()
        self.z_dim = z_dim 
        self.channel_dim = channel_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_dim, 4, 2, 1),            
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1) # TODO: check this out
        output = self.main(x)
        output = torch.tanh(output)
        return output

    # TODO: there are more methods: loss(..) and predict(...) - check if these are needed