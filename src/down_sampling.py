import logging
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="logs/encoder.log",
)


class Encoder(nn.Module):
    """
    An Encoder module designed for convolutional neural network architectures, especially useful in generative models and autoencoders. This module constructs a convolutional encoder block, optionally incorporating an activation layer and a normalization layer.

    ## Parameters

    | Parameter      | Type    | Description                                           | Default |
    |----------------|---------|-------------------------------------------------------|---------|
    | in_channels    | int     | Number of channels in the input tensor.               | None    |
    | out_channels   | int     | Number of channels produced by the convolution.       | None    |
    | kernel_size    | int     | Size of the convolving kernel.                        | None    |
    | stride         | int     | Stride of the convolution.                            | None    |
    | padding        | int     | Zero-padding added to both sides of the input.        | None    |
    | use_leakyReLU  | bool    | Whether to include a LeakyReLU activation layer.      | None    |
    | use_norm       | bool    | Whether to include a batch normalization layer.       | None    |

    ## Attributes

    | Attribute | Type             | Description                                         |
    |-----------|------------------|-----------------------------------------------------|
    | model     | `nn.Sequential`  | The sequential model comprising the encoder block. |

    ## Methods

    - `encoder_block()`: Constructs the encoder block with the specified layers.
    - `forward(x)`: Defines the forward pass of the Encoder.

    ## Example Usage

    ```python
    # Example instantiation of the Encoder class
    netE = Encoder(
        in_channels=3,
        out_channels=64,
        kernel_size=4,
        stride=2,
        padding=1,
        use_leakyReLU=True,
        use_norm=True
    )

    # Example forward pass
    x = torch.randn(1, 3, 256, 256)  # assuming a sample input tensor
    output = netE.forward(x)
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
        """
        Constructs the encoder block by sequentially adding a convolutional layer followed optionally by an activation layer (LeakyReLU) and a normalization layer (BatchNorm2d), as specified during the module's initialization.

        Returns:
            nn.Sequential: A sequential container of the constructed encoder block layers.
        """
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
        """
        Defines the forward pass of the Encoder, passing the input tensor `x` through the encoder block.

        Parameters:
            x (Tensor): The input tensor to be encoded.

        Returns:
            Tensor: The output tensor after encoding.
        """
        return self.model(x)


if __name__ == "__main__":
    netE = Encoder()
