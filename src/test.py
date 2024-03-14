import sys
import os
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)
sys.path.append("src/")

from generator import Generator
from utils import device_init, load_pickle
from config import LAST_CHECKPOINTS, PROCESSED_DATA_PATH


class Test:
    def __init__(self, num_samples=20, device="mps"):
        self.num_samples = num_samples
        self.device = device_init(device=device)
        self.images = list()

    def select_best_model(self):
        if os.path.exists(LAST_CHECKPOINTS):
            return torch.load(
                os.path.join(
                    "/Users/shahmuhammadraditrahman/Desktop/pix2pix/checkpoints/best_model/best_model.pth"
                )
            )
        else:
            raise Exception("No checkpoints found".capitalize())

    def plot_data_comparison(self, netG=None):
        if os.path.exists(PROCESSED_DATA_PATH):
            dataloader = load_pickle(
                os.path.join(PROCESSED_DATA_PATH, "dataloader.pkl")
            )
            for _ in range(self.num_samples):
                data, label = next(iter(dataloader))
                self.images.append(data)

            plt.figure(figsize=(10, 5))

            for index, image in enumerate(self.images):
                inputs = image[:, :, :, :256].to(self.device)
                inputs = inputs.squeeze().permute(1, 2, 0)
                targets = image[:, :, :, 256:].to(self.device)
                targets = targets.squeeze().permute(1, 2, 0)

                inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
                targets = (targets - targets.min()) / (targets.max() - targets.min())

                output = netG(inputs)

                output = (output - output.min()) / (output.max() - output.min())

                plt.subplot(4 * 2, 5 * 2, 2 * index + 2)
                plt.imshow(targets)
                plt.title("Ground Truth".capitalize())
                plt.axis("off")

                plt.subplot(4 * 2, 5 * 2, 2 * index + 2)
                plt.imshow(output)
                plt.title("Generated".capitalize())
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        else:
            raise Exception("No processed data found".capitalize())

    def test(self):
        netG = Generator().to(self.device)
        netG.load_state_dict(self.select_best_model())

        self.plot_data_comparison(netG=netG)


if __name__ == "__main__":
    plot = Test()
    plot.test()


# device = device_init(device="mps")

# netG = Generator().to(device)

# model = torch.load("./checkpoints/last_model/last_netG.pth")
# netG.load_state_dict(model)


# dataloader = load_pickle(path="./data/processed/dataloader.pkl")

# images = list()

# for _ in range(20):
#     data, label = next(iter(dataloader))
#     images.append(data)

# plt.figure(figsize=(25, 15))  # Adjusted figsize to accommodate two images per row.

# for index, image in enumerate(images):
#     # Assuming image is divided into input and target parts.
#     input_image = image[:, :, :, :256].to(device)
#     target = image[:, :, :, 256:].to(device)

#     # Process the target image.
#     target_processed = target.squeeze().permute(1, 2, 0).cpu().detach().numpy()
#     target_processed = (target_processed - target_processed.min()) / (
#         target_processed.max() - target_processed.min()
#     )

#     # Generate an image using netG.
#     generated_image = netG(input_image)
#     generated_image_processed = (
#         generated_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
#     )
#     generated_image_processed = (
#         generated_image_processed - generated_image_processed.min()
#     ) / (generated_image_processed.max() - generated_image_processed.min())

#     # Display the target image.
#     plt.subplot(4, 10, 2 * index + 1)
#     plt.imshow(target_processed)
#     plt.title("Ground truth".capitalize())
#     plt.axis("off")

#     # Display the generated image.
#     plt.subplot(4, 10, 2 * index + 2)
#     plt.imshow(generated_image_processed)
#     plt.title("Generated".capitalize())
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
