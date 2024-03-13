import sys
import os
import yaml
import joblib
import torch

sys.path.append("src/")


def params():
    with open("./default_params.yml", "r") as file:
        return yaml.safe_load(file)


def load_pickle(path):
    if os.path.exists(path):
        return joblib.load(filename=path)
    else:
        raise Exception("Path is not found".capitalize())


def weight_init(m):
    if m is not None:
        classname = m.__class__.__name__

        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    else:
        raise ValueError("Model is not defined".capitalize())
