import logging
import torch
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/decoder.log",
)


class Decoder(nn.Module):
    """
    A Decoder module designed for use in neural network architectures, specifically within generative models like GANs or autoencoders. This module creates a convolutional decoder block that can optionally include activation and normalization layers.

    ## Parameters

    | Parameter      | Type    | Description                                           | Default |
    |----------------|---------|-------------------------------------------------------|---------|
    | in_channels    | int     | Number of channels in the input tensor.               | None    |
    | out_channels   | int     | Number of channels produced by the convolution.       | None    |
    | kernel_size    | int     | Size of the convolving kernel.                        | None    |
    | stride         | int     | Stride of the convolution.                            | None    |
    | padding        | int     | Zero-padding added to both sides of the input.        | None    |
    | use_leakyReLU  | bool    | Whether to include a ReLU activation layer.           | None    |
    | use_norm       | bool    | Whether to include a batch normalization layer.       | None    |

    ## Attributes

    | Attribute | Type             | Description                                         |
    |-----------|------------------|-----------------------------------------------------|
    | model     | `nn.Sequential`  | The sequential model comprising the decoder block. |

    ## Methods

    - `decoder_block()`: Constructs the decoder block with the specified layers.
    - `forward(encoder, x)`: Defines the forward pass of the Decoder.

    ## Example Usage

    ```python
    # Example instantiation of the Decoder class
    netD = Decoder(
        in_channels=512,
        out_channels=256,
        kernel_size=4,
        stride=2,
        padding=1,
        use_leakyReLU=True,
        use_norm=True
    )

    # Example forward pass
    encoder_output = some_encoder_output  # assuming shape [batch_size, in_channels, H, W]
    x = some_input_tensor  # assuming shape [batch_size, in_channels, H, W]
    output = netD.forward(encoder_output, x)
    ```
    """

    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        stride=None,
        padding=None,
        use_leakyReLU=None,
        use_norm=None,
        use_dropout=None,
    ):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_leakyReLU = use_leakyReLU
        self.use_norm = use_norm
        self.use_dropout = use_dropout

        self.model = self.decoder_block()

    def decoder_block(self):
        """
        Constructs the decoder block comprising a convolutional layer optionally followed by activation and normalization layers, based on the initialization parameters.

        Returns:
            nn.Sequential: A sequential container of the constructed decoder block layers.
        """
        layers = OrderedDict()
        layers["decoder"] = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=False,
        )
        if self.use_leakyReLU:
            layers["ReLU"] = nn.ReLU(inplace=True)
        if self.use_norm:
            layers["batch_norm"] = nn.BatchNorm2d(self.out_channels)
        if self.use_dropout:
            layers["dropout"] = nn.Dropout2d(p=0.5)

        return nn.Sequential(layers)

    def forward(self, x, skip_info):
        """
        Defines the forward pass of the Decoder. It concatenates the encoder output with an input tensor along the channel dimension before passing through the decoder block.

        Parameters:
            encoder (Tensor): The output tensor from the corresponding encoder block.
            x (Tensor): The input tensor to be concatenated with the encoder output.

        Returns:
            Tensor: The output tensor after decoding.
        """
        x = self.model(x)
        x = torch.cat((x, skip_info), 1)
        return x


if __name__ == "__main__":
    netD = Decoder()
