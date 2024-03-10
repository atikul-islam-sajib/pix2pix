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
    """
    Implements a Generator model for image-to-image translation tasks, specifically designed to incorporate a series of encoding and decoding blocks. This class leverages custom Encoder and Decoder modules to progressively downsample and then upsample an input image, potentially for tasks such as image segmentation, style transfer, or data augmentation.

    ## Attributes

    The Generator is composed of several Encoder (`netG1` to `netG7`) and Decoder (`netD1` to `netD7`) blocks. Each block is responsible for a stage of down-sampling or up-sampling and is parameterized by its own set of hyperparameters including the number of input and output channels, kernel size, stride, and padding.

    | Attribute | Type      | Description                                            |
    |-----------|-----------|--------------------------------------------------------|
    | netG1-netG7 | `Encoder` | Encoder blocks for down-sampling the input image.      |
    | netD1-netD7 | `Decoder` | Decoder blocks for up-sampling the encoded representation back to the target resolution. |

    ## Methods

    - `forward(x)`: Defines the forward pass of the Generator with input `x`.

    ## Example Usage

    ```python
    # Initialize the Generator model.
    gen = Generator()

    # Assuming `x` is your input tensor with the appropriate dimensions (e.g., [batch_size, channels, height, width]).
    # Generate the transformed image.
    transformed_image = gen(x)
    ```

    This Generator model can be adapted and extended depending on the specific requirements of your image-to-image translation task. Its modular design allows for customization of the encoding and decoding stages.
    """

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
        """
        Performs the forward pass of the Generator, sequentially passing the input through each encoder block, reaching a latent representation, and then progressively up-sampling and merging this representation through the decoder blocks to produce the final output image.

        Parameters:
            x (Tensor): The input tensor to the Generator model, typically an image or batch of images with shape `[batch_size, channels, height, width]`.

        Returns:
            Tensor: The transformed output tensor, having the same shape as the input tensor.
        """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generator model for image to image translation".title()
    )
    parser.add_argument(
        "--model", action="store_true", help="Define the model".capitalize()
    )

    args = parser.parse_args()

    if args.model:
        netG = Generator()

        logging.info(netG)
    else:
        raise Exception("Model not defined".capitalize())
