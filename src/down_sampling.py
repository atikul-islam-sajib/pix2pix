import logging
import argparse
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/encoder.log",
)


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        stride=None,
        padding=None,
        use_leakyReLU=None,
        use_norm=None,
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_leakyReLU = use_leakyReLU
        self.use_norm = use_norm
        self.model = self.encoder_block()

    def encoder_block(self):
        layers = OrderedDict()
        layers["encoder"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=False,
        )
        if self.use_leakyReLU:
            layers["leakyReLU"] = nn.LeakyReLU(0.2, inplace=True)
        if self.use_norm:
            layers["batch_norm"] = nn.BatchNorm2d(self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    netE = Encoder()
