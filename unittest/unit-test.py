import sys
import unittest
import torch

sys.path.append("src/")

from utils import load_pickle
from generator import Generator
from discriminator import Discriminator


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = load_pickle(path="./data/processed/dataloader.pkl")
        self.netG = Generator()
        self.netD = Discriminator()
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

    def test_generator_params(self):
        self.assertEquals(
            sum(params.numel() for params in self.netG.parameters()), 41828992
        )

    def test_generate_image_size(self):
        self.assertEqual(self.netG(self.data).size(), torch.Size([1, 3, 256, 512]))

    def test_discriminator_params(self):
        self.assertEquals(
            sum(params.numel() for params in self.netD.parameters()), 2769601
        )

    def test_discriminator_image_size(self):
        self.assertEqual(self.netD(self.inputs).size(), torch.Size([1, 1, 30, 30]))


if __name__ == "__main__":
    unittest.main()
