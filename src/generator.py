import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/generator.log",
)


class Generator(nn.Module):
    """
    Implements a Generator class for image-to-image translation tasks, following the architecture commonly used in Generative Adversarial Networks (GANs). This class builds a deep convolutional neural network with specific layers designed for encoding and decoding images.

    ## Parameters

    This class does not take parameters in its constructor.

    ## Attributes

    | Attribute | Type               | Description                                             |
    |-----------|--------------------|---------------------------------------------------------|
    | encoder1  | `nn.Conv2d`        | Convolutional layer for initial encoding.               |
    | encoder2  | `nn.Sequential`    | Second encoding layer with Conv2d, LeakyReLU, and BatchNorm2d. |
    | encoder3  | `nn.Sequential`    | Third encoding layer with similar structure.            |
    | ...       |                    |                                                         |
    | decoder1  | `nn.Sequential`    | First decoding layer with ConvTranspose2d, ReLU, BatchNorm2d, and optional Dropout. |
    | decoder2  | `nn.Sequential`    | Second decoding layer with similar structure.           |
    | ...       |                    |                                                         |

    ## Methods

    - `forward(x)`: Defines the forward pass of the generator with input `x`. Concatenates encoder outputs with corresponding decoder layers to form the latent space and generate the output image.

    ## Example Usage

    ```python
    # Assuming you have PyTorch and necessary libraries installed.
    # Create an instance of the Generator.
    netG = Generator()

    # Assuming `x` is a batch of images with shape [batch_size, 3, height, width].
    # Generate transformed images.
    transformed_images = netG(x)
    ```
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )
        self.encoder7 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, x):
        """
        Defines the forward pass of the Generator with the given input `x`.

        ## Parameters

        | Parameter | Type      | Description                                  |
        |-----------|-----------|----------------------------------------------|
        | x         | `Tensor`  | The input tensor with shape [N, C, H, W].    |

        ## Returns

        | Return    | Type      | Description                                  |
        |-----------|-----------|----------------------------------------------|
        | output    | `Tensor`  | The transformed tensor with shape [N, C, H, W]. |

        ## Example

        ```python
        # Assuming `x` is your input tensor with the correct shape.
        output = netG(x)
        ```
        """
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        encoder6 = self.encoder6(encoder5)
        latent_space = self.encoder7(encoder6)

        decoder1 = torch.cat((encoder6, self.decoder1(latent_space)), dim=1)
        decoder2 = torch.cat((encoder5, self.decoder2(decoder1)), dim=1)
        decoder3 = torch.cat((encoder4, self.decoder3(decoder2)), dim=1)
        decoder4 = torch.cat((encoder3, self.decoder4(decoder3)), dim=1)
        decoder5 = torch.cat((encoder2, self.decoder5(decoder4)), dim=1)
        decoder6 = torch.cat((encoder1, self.decoder6(decoder5)), dim=1)

        return self.decoder7(decoder6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Generator model for image to image translation".title()
    )

    parser.add_argument(
        "--model", action="store_true", help="Define the model".capitalize()
    )

    args = parser.parse_args()

    if args.model:
        netG = Generator()
        logging.info(netG)
    else:
        raise Exception("Define the model".capitalize())
