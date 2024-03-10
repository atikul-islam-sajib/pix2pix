import sys
import unittest
import torch

sys.path.append("src/")

from utils import load_pickle
from generator import Generator
from discriminator import Discriminator
from generator1 import Generator


class UnitTest(unittest.TestCase):
    """
    This module contains unit tests for the Generative Adversarial Network (GAN) components, including the Generator and Discriminator models, as well as data loading functionalities. These tests ensure that the models and data handling processes perform as expected.

    ## Classes

    - `UnitTest`: A suite of tests for validating the integrity and expected behavior of the GAN components.

    ## Usage

    To run these tests, navigate to the directory containing this script and execute:

    ```bash
    python -m unittest <name_of_this_test_file>.py


    ## Methods

    - `setUp`: Initializes the components and data required for the tests.
    - `test_quantity_of_data`: Verifies the total quantity of data items in the dataloader.
    - `test_size_of_data`: Checks the size of a data batch from the dataloader.
    - `test_generator_params`: Confirms the total number of parameters in the Generator model.
    - `test_generate_image_size`: Ensures the output size of the Generator model is as expected.
    - `test_discriminator_params`: Validates the total number of parameters in the Discriminator model.
    - `test_discriminator_image_size`: Asserts the output size of the Discriminator model is correct.
    """

    def setUp(self):
        self.dataloader = load_pickle(path="./data/processed/dataloader.pkl")
        # self.netG = Generator()
        self.netD = Discriminator()
        self.netG1 = Generator()
        self.data = torch.randn(1, 3, 256, 512)
        self.image = torch.randn(1, 3, 256, 512)
        self.data1 = self.image[:, :, :, :256]
        self.data2 = self.image[:, :, :, 256:]

        self.inputs = torch.cat((self.data1, self.data2), dim=1)

    def test_quantity_of_data(self):
        self.assertEqual(sum(data.size(0) for data, _ in self.dataloader), 2194)

    def test_size_of_data(self):
        data, _ = next(iter(self.dataloader))
        self.assertEqual(data.size(), torch.Size([1, 3, 256, 512]))

    # def test_generator_params(self):
    #     self.assertEquals(
    #         sum(params.numel() for params in self.netG.parameters()), 41828992
    #     )

    # def test_generate_image_size(self):
    #     self.assertEqual(self.netG(self.data).size(), torch.Size([1, 3, 256, 512]))

    def test_discriminator_params(self):
        self.assertEquals(
            sum(params.numel() for params in self.netD.parameters()), 2768640
        )

    def test_discriminator_image_size(self):
        self.assertEqual(self.netD(self.inputs).size(), torch.Size([1, 1, 30, 30]))

    def test_generator1_params(self):
        self.assertEquals(
            sum(params.numel() for params in self.netG1.parameters()), 41828992
        )

    def test_generate1_image_size(self):
        self.assertEqual(self.netG1(self.data).size(), torch.Size([1, 3, 256, 512]))


if __name__ == "__main__":
    unittest.main()
