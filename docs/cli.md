# Pix2Pix Image Translation - cli

This project implements the Pix2Pix generative adversarial network (GAN) for image-to-image translation tasks. It provides tools for training and testing the model using custom datasets. This README outlines how to set up and use the project.

## Features

- Train Pix2Pix models on custom datasets.
- Test trained models to generate image translations.
- Normalize input data.
- Configure training parameters such as epochs, learning rate, and lambda values.
- Support for multiple devices including CPU, GPU, and Apple Silicon (MPS).
- Generate samples and display outputs during testing.

### Training Command Comparison

| Device Type         | Command                                                                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                 | `python main.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device cpu --display True`  |
| GPU                 | `python main.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device cuda --display True` |
| Apple Silicon (MPS) | `python main.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device mps --display True`  |

### Full Command Options in Detail

Here's a detailed breakdown of the command options for training and testing the Pix2Pix model:

- `--dataset <path>`: Specifies the path to the zip file containing your dataset.
- `--normalized <True|False>`: Indicates whether the data should be normalized. Default is True.
- `--epochs <number>`: Sets the number of training epochs. Default is 20.
- `--lr <value>`: Defines the learning rate for the optimizer. Default is 0.0002.
- `--beta1 <value>`: Sets the Beta1 hyperparameter for the Adam optimizer. Default is 0.5.
- `--lambda_value <value>`: Specifies the lambda value for loss calculation. Default is 100.
- `--device <cuda|cpu|mps>`: Chooses the device to run the model on. Default is 'mps'.
- `--display <True|False>`: Determines whether to display the output during training/testing. Default is True.
- `--samples <number>`: Decides the number of samples to use for plotting during testing. Default is 20.
- `--train`: Flag to initiate training mode.
- `--test`: Flag to initiate testing mode.

This tabular form and detailed options guide should provide a clear understanding of how to set up and execute the Pix2Pix model training on different devices, as well as how to adjust various training parameters according to your needs.
