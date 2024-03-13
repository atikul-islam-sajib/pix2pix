import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src/")

from generator import Generator
from utils import device_init, load_pickle

device = device_init(device="mps")

netG = Generator().to(device)

model = torch.load("./checkpoints/last_model/last_netG.pth")
netG.load_state_dict(model)


dataloader = load_pickle(path="./data/processed/dataloader.pkl")

images = list()

for _ in range(20):
    data, label = next(iter(dataloader))
    images.append(data)

plt.figure(figsize=(25, 15))  # Adjusted figsize to accommodate two images per row.

for index, image in enumerate(images):
    # Assuming image is divided into input and target parts.
    input_image = image[:, :, :, :256].to(device)
    target = image[:, :, :, 256:].to(device)

    # Process the target image.
    target_processed = target.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    target_processed = (target_processed - target_processed.min()) / (
        target_processed.max() - target_processed.min()
    )

    # Generate an image using netG.
    generated_image = netG(input_image)
    generated_image_processed = (
        generated_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    )
    generated_image_processed = (
        generated_image_processed - generated_image_processed.min()
    ) / (generated_image_processed.max() - generated_image_processed.min())

    # Display the target image.
    plt.subplot(4, 10, 2 * index + 1)
    plt.imshow(target_processed)
    plt.title("Ground truth".capitalize())
    plt.axis("off")

    # Display the generated image.
    plt.subplot(4, 10, 2 * index + 2)
    plt.imshow(generated_image_processed)
    plt.title("Generated".capitalize())
    plt.axis("off")

plt.tight_layout()
plt.show()
