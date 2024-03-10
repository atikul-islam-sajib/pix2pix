import sys
import logging
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/generator1.log",
)
sys.path.append("src/")

from down_sampling import Encoder
from up_sampling import Decoder


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.netG1 = Encoder(3, 64, 4, 2, 1, False, False)
        self.netG2 = Encoder(64, 128, 4, 2, 1, True, True)
        self.netG3 = Encoder(128, 256, 4, 2, 1, True, True)
        self.netG4 = Encoder(256, 512, 4, 2, 1, True, True)
        self.netG5 = Encoder(512, 512, 4, 2, 1, True, True)
        self.netG6 = Encoder(512, 512, 4, 2, 1, True, True)
        self.netG7 = Encoder(512, 512, 4, 2, 1, True, False)

        self.netD1 = Decoder(512, 512, 4, 2, 1, False, True)
        self.netD2 = Decoder(1024, 512, 4, 2, 1, True, True)
        self.netD3 = Decoder(1024, 512, 4, 2, 1, True, True)
        self.netD4 = Decoder(1024, 256, 4, 2, 1, True, True)
        self.netD5 = Decoder(512, 128, 4, 2, 1, True, True)
        self.netD6 = Decoder(256, 64, 4, 2, 1, True, True)
        self.netD7 = Decoder(128, 3, 4, 2, 1, True, False)

    def forward(self, x):
        x1 = self.netG1(x)
        x2 = self.netG2(x1)
        x3 = self.netG3(x2)
        x4 = self.netG4(x3)
        x5 = self.netG5(x4)
        x6 = self.netG6(x5)

        latent_space = self.netG7(x6)

        x = torch.cat((x6, self.netD1(latent_space)), 1)
        x = torch.cat((x5, self.netD2(x)), 1)
        x = torch.cat((x4, self.netD3(x)), 1)
        x = torch.cat((x3, self.netD4(x)), 1)
        x = torch.cat((x2, self.netD5(x)), 1)
        x = torch.cat((x1, self.netD6(x)), 1)

        x = self.de7(x)

        return nn.Sigmoid(x)


gen = Generator()

print(sum(params.numel() for params in gen.parameters()))
