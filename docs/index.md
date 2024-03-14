# Pix2Pix Image Translation

## Overview

Pix2Pix is a machine learning project focused on translating images from one domain to another using Conditional Generative Adversarial Networks (cGANs). This tool is designed to work with a wide range of datasets, including custom datasets, for various image translation tasks such as transforming sketch images into photo-realistic images, or black and white images into color.

## Features

- Built with PyTorch, leveraging the power of deep learning for image translation.
- Easy-to-use scripts for training and generating synthetic images.
- Command Line Interface (CLI) for straightforward interaction.
- Supports custom data loaders for diverse datasets.
- Adjustable training parameters for model fine-tuning.

## Installation

Clone the repository to your local machine:

```
git clone https://github.com/atikul-islam-sajib/pix2pix.git
cd pix2pix
```

### Install Dependencies

```
pip install -r requirements.txt
```

## Usage

The project supports both training and testing modes. Below are example commands and their explanations.

### Training

To start training the Pix2Pix model:

```bash
python cli.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device cuda --display True
```

### Testing

To test the Pix2Pix model:

```bash
python cli.py --test --samples 20 --device cuda
```

### CLI Options

| Argument         | Description                                     | Type    | Default |
| ---------------- | ----------------------------------------------- | ------- | ------- |
| `--dataset`      | Path to the dataset zip file.                   | `str`   | None    |
| `--epochs`       | Number of epochs for training.                  | `int`   | 200     |
| `--lr`           | Learning rate for the optimizer.                | `float` | 0.0002  |
| `--beta1`        | Beta1 hyperparameter for the Adam optimizer.    | `float` | 0.5     |
| `--lambda_value` | Lambda weight for L1 loss.                      | `float` | 100     |
| `--device`       | Device to run the model ('cuda', 'cpu', 'mps'). | `str`   | 'cuda'  |
| `--samples`      | Number of samples to generate for testing.      | `int`   | 20      |
| `--train`        | Flag to initiate the training process.          | -       | -       |
| `--test`         | Flag to initiate the testing process.           | -       | -       |

## Training and Generating Images

### Training the Model

Train your model on any device by adapting the `--device` option accordingly:

| Device | Command                                                                                                                                        |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| CUDA   | `python cli.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device cuda --display True` |
| CPU    | `python cli.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device cpu --display True`  |
| MPS    | `python cli.py --train --dataset path/to/your/dataset.zip --epochs 20 --lr 0.0002 --beta1 0.5 --lambda_value 100 --device mps --display True`  |

### Testing the Model

Generate images with your trained model:

| Device | Command                                                                           |
| ------ | --------------------------------------------------------------------------------- |
| CUDA   | `python test.py --test --model_path path/to/model.pth --samples 20 --device cuda` |
| CPU    | `python test.py --test --model_path path/to/model.pth --samples 20 --device cpu`  |
| MPS    | `python test.py --test --model_path path/to/model.pth --samples 20 --device mps`  |

## Contributing

Your contributions are welcome! Please follow the standard fork-pull request workflow to submit your improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

Thank you to all the contributors of the Pix2Pix project. Special thanks to the authors of the original Pix2Pix paper for their groundbreaking work in image-to-image translation.

## Contact

For questions or suggestions, please contact [atikulislamsajib137@gmail.com].
