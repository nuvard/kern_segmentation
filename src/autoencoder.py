
from __future__ import print_function, division
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from torchvision.utils import save_image

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 3, 1, stride=1, padding=0),  # b, 16, 10, 10
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 6, 1, stride=1, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            #nn.ConvTranspose2d(3, 6, 1, stride=1, padding=0),  # b, 1, 28, 28
        )
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        x_1 = self.encoder(x)
        x = self.decoder(x_1)
        return (x_1, x)
