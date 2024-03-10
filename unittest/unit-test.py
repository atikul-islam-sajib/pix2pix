import sys
import unittest
import torch

sys.path.append("src/")

from utils import load_pickle


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = load_pickle(path="./data/processed/dataloader.pkl")

    def test_quantity_of_data(self):
        self.assertEqual(sum(data.size(0) for data, _ in self.dataloader), 2194)

    def test_size_of_data(self):
        data, _ = next(iter(self.dataloader))
        self.assertEqual(data.size(), torch.Size([1, 3, 256, 512]))


if __name__ == "__main__":
    unittest.main()
