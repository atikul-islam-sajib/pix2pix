import sys
import os
import yaml
import joblib
import torch
import warnings

sys.path.append("src/")
from config import TRAIN_CHECKPOINTS, LAST_CHECKPOINTS


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


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def ignore_warnings():
    warnings.filterwarnings("ignore")


def saved_config(config_file=None, filename=None):
    if config_file is not None and filename is not None:
        with open(os.path.join(filename, ".yml"), "w") as file:
            yaml.safe_dump(config_file, file)
    else:
        raise ValueError("Define the arguments properly".capitalize())


def clean():
    if os.path.exists(TRAIN_CHECKPOINTS) and os.path.exists(LAST_CHECKPOINTS):
        for file in os.listdir(TRAIN_CHECKPOINTS):
            os.remove(os.path.join(TRAIN_CHECKPOINTS, file))

        for file in os.listdir(LAST_CHECKPOINTS):
            os.remove(os.path.join(LAST_CHECKPOINTS, file))
