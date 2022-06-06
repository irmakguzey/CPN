import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # print("mean: {}".format(x.mean()))
        print(x.shape)
        return x

# Class to learn current embeddings
class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()

        # print('inside Encoders init')
        self.z_dim = z_dim
        self.model = nn.Sequential(
            # PrintSize(),
            nn.Conv2d(channel_dim, 64, 8, 4, 1),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.4),
            # PrintSize(),

            nn.Conv2d(64, 32, 8, 4, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            # 64 x 32 x 32
            # PrintSize(),

            nn.Conv2d(32, 16, 8, 4, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(16, 16, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),

            # nn.Conv2d(128, 128, 8, 4, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            # PrintSize(),

            # nn.Conv2d(128, 128, 8, 4, 1),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(64, 64, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),

            # nn.Conv2d(64, 128, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # # 128 x 16 x 16
            # # PrintSize(),

            # nn.Conv2d(128, 128, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # # Option 1: 256 x 8 x 8
            # # PrintSize(),

            # nn.Conv2d(128, 128, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # # 256 x 4 x 4
            # # PrintSize(),

            # nn.Conv2d(128, 128, 4, 1, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # # PrintSize()

            # nn.Conv2d(128, 64, 4, 2, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # PrintSize()
        )

        self.out = nn.Linear(16 * 3 * 3, z_dim)
        # self.out = nn.Sequential(
        #     nn.Linear(128 * 7 * 10, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, z_dim)
        # )

        # print('done with Encoder')

    def forward(self, x):
        x = self.model(x) # This works perfectly fine
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x

# Model to predict next embeddings given an embedding and an action
class Transition(nn.Module):
    def __init__(self, z_dim, action_dim):
        super().__init__()

        # print('inside Transition init')

        self.z_dim = z_dim 
        hidden_dim = 64
        self.model = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
        # print('done with Transition')

    def forward(self, z, a):
        x = torch.cat((z,a), dim=-1)
        x = self.model(x)
        return x