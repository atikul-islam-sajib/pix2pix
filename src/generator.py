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
        self.encoder_layer1 = Encoder(3, 64, 4, 2, 1, False, False)
        self.encoder_layer2 = Encoder(64, 128, 4, 2, 1, True, True)
        self.encoder_layer3 = Encoder(128, 256, 4, 2, 1, True, True)
        self.encoder_layer4 = Encoder(256, 512, 4, 2, 1, True, True)
        self.encoder_layer5 = Encoder(512, 512, 4, 2, 1, True, True)
        self.encoder_layer6 = Encoder(512, 512, 4, 2, 1, True, True)
        self.encoder_layer7 = Encoder(512, 512, 4, 2, 1, True, False)

        # Decoding layers: progressively upsample the feature representation to reconstruct the image
        self.decoder_layer1 = Decoder(512, 512, 4, 2, 1, True, True, True)
        self.decoder_layer2 = Decoder(1024, 512, 4, 2, 1, True, True, True)
        self.decoder_layer3 = Decoder(1024, 512, 4, 2, 1, True, True, True)
        self.decoder_layer4 = Decoder(1024, 256, 4, 2, 1, True, True, False)
        self.decoder_layer5 = Decoder(512, 128, 4, 2, 1, True, True, False)
        self.decoder_layer6 = Decoder(256, 64, 4, 2, 1, True, True, False)

        # Output layer to generate the final image
        self.output_layer = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)

    def forward(self, x):
        """
        Performs the forward pass of the Generator, sequentially passing the input through each encoder block, reaching a latent representation, and then progressively up-sampling and merging this representation through the decoder blocks to produce the final output image.

        Parameters:
            x (Tensor): The input tensor to the Generator model, typically an image or batch of images with shape `[batch_size, channels, height, width]`.

        Returns:
            Tensor: The transformed output tensor, having the same shape as the input tensor.
        """
        enc_layer1_out = self.encoder_layer1(x)
        enc_layer2_out = self.encoder_layer2(enc_layer1_out)
        enc_layer3_out = self.encoder_layer3(enc_layer2_out)
        enc_layer4_out = self.encoder_layer4(enc_layer3_out)
        enc_layer5_out = self.encoder_layer5(enc_layer4_out)
        enc_layer6_out = self.encoder_layer6(enc_layer5_out)
        encoded_out = self.encoder_layer7(enc_layer6_out)

        # Decoder forward pass with skip connections
        dec_layer1_out = self.decoder_layer1(encoded_out, enc_layer6_out)
        dec_layer2_out = self.decoder_layer2(dec_layer1_out, enc_layer5_out)
        dec_layer3_out = self.decoder_layer3(dec_layer2_out, enc_layer4_out)
        dec_layer4_out = self.decoder_layer4(dec_layer3_out, enc_layer3_out)
        dec_layer5_out = self.decoder_layer5(dec_layer4_out, enc_layer2_out)
        dec_layer6_out = self.decoder_layer6(dec_layer5_out, enc_layer1_out)

        # Final output
        final_output = self.output_layer(dec_layer6_out)
        return torch.sigmoid(final_output)


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
