import sys
import os
import logging
import argparse
import torch
import imageio
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)
sys.path.append("src/")

from generator import Generator
from utils import device_init, load_pickle, ignore_warnings
from config import (
    LAST_CHECKPOINTS,
    PROCESSED_DATA_PATH,
    TRAIN_IMAGES,
    GIF_PATH,
    TEST_IMAGES,
)


class Test:
    """
    A class for testing the generative model by comparing generated images to their corresponding targets,
    and optionally generating a GIF from training images.

    This class initializes with a specified number of samples and device, selects the best model based on
    saved checkpoints, compares generated images against real images, and creates a GIF from the training images.

    Parameters
    ----------
    | Parameter    | Type | Description                                                  | Default |
    |--------------|------|--------------------------------------------------------------|---------|
    | num_samples  | int  | The number of samples to generate for comparison.            | 20      |
    | device       | str  | The device to run the model on ('cuda', 'cpu', 'mps').       | 'mps'   |



    Attributes
    ----------
    | Attribute | Type | Description                                          |
    |-----------|------|------------------------------------------------------|
    | images    | list | Stores images for plotting or GIF generation.        |



    Methods
    -------
    | Method               | Description                                                                                   |
    |----------------------|-----------------------------------------------------------------------------------------------|
    | select_best_model()  | Selects and loads the best model from saved checkpoints.                                      |
    | plot_data_comparison | Plots a comparison between real images, target images, and images generated by the model.     |
    |                      | (`netG=None`) Accepts an optional generator model for generating images.                      |
    | generate_gif()       | Generates a GIF from images stored in a specified directory.                                  |
    | test()               | Executes the testing process, including model loading, data plotting, and GIF generation.     |


    Examples
    --------
    To use this class, initialize with the desired number of samples and the device. Then, call the `test` method
    to perform the test process:

        >>> test_instance = Test(num_samples=20, device='mps')
        >>> test_instance.test()
    """

    def __init__(self, num_samples=20, device="mps"):
        self.num_samples = num_samples
        self.device = device_init(device=device)
        self.images = list()

    def select_best_model(self):
        if os.path.exists(LAST_CHECKPOINTS):
            return torch.load(
                os.path.join(os.path.join(LAST_CHECKPOINTS, "last_netG.pth"))
            )
        else:
            raise Exception("No checkpoints found".capitalize())

    def plot_data_comparison(self, netG=None):
        """
        Plots a comparison between real images, target images, and images generated by the model.

        Parameters
        ----------
        | Parameter | Type            | Description                                                              | Default |
        |-----------|-----------------|--------------------------------------------------------------------------|---------|
        | netG      | torch.nn.Module | The generator model to use for generating images. Assumes model is ready if not provided. | None    |
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            dataloader = load_pickle(
                os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )
            for _ in range(self.num_samples):
                data, label = next(iter(dataloader))
                self.images.append(data)

            plt.figure(figsize=(40, 20))

            for index, image in enumerate(self.images):
                inputs = image[:, :, :, :256].to(self.device)
                targets = image[:, :, :, 256:].to(self.device)
                targets_cpu = targets.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                inputs_normalized = (inputs - inputs.min()) / (
                    inputs.max() - inputs.min()
                )

                output = netG(inputs_normalized).cpu().detach()
                output = output.squeeze().permute(1, 2, 0).numpy()
                output_normalized = (output - output.min()) / (
                    output.max() - output.min()
                )

                plt.subplot(4 * 3, 5 * 3, 3 * index + 1)
                plt.imshow(
                    inputs_normalized.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )
                plt.title("Real".capitalize())
                plt.axis("off")

                plt.subplot(4 * 3, 5 * 3, 3 * index + 2)
                plt.imshow(targets_cpu)
                plt.title("Target".capitalize())
                plt.axis("off")

                plt.subplot(4 * 3, 5 * 3, 3 * index + 3)
                plt.imshow(output_normalized)
                plt.title("Generate".capitalize())
                plt.axis("off")

            plt.tight_layout()
            try:
                plt.savefig(os.path.join(TEST_IMAGES, "test.png"))
            except Exception as e:
                logging.exception("Test image path is not defined".capitalize())
            else:
                plt.show()

        else:
            raise Exception("No processed data found".capitalize())

    def generate_gif(self):
        if os.path.exists(GIF_PATH):
            images = [
                imageio.imread(os.path.join(TRAIN_IMAGES, image))
                for image in os.listdir(os.path.join(TRAIN_IMAGES))
            ]
            imageio.mimsave(
                os.path.join(GIF_PATH, "result.gif"), images, "GIF", duration=20
            )
        else:
            raise Exception("No processed data found".capitalize())

    def test(self):
        try:
            ignore_warnings()
            netG = Generator().to(self.device)
            netG.load_state_dict(self.select_best_model())
        except Exception as e:
            print("Exception caught in the section - ".capitalize(), e)
        else:
            self.plot_data_comparison(netG=netG)
        finally:
            self.generate_gif()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model".title())
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples that is used for plotting".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to be used for training".capitalize(),
    )

    args = parser.parse_args()

    if args.device and args.samples:
        logging.info("Testing the model".capitalize())

        plot = Test(num_samples=args.samples, device=args.device)
        plot.test()

        logging.info("Testing completed".capitalize())
    else:
        raise Exception("Invalid arguments".capitalize())
