import sys
import os
import yaml
import joblib
import torch
import warnings

sys.path.append("src/")
from config import TRAIN_CHECKPOINTS, LAST_CHECKPOINTS, TEST_IMAGES


def params():
    """
    Load parameters from a YAML file named 'default_params.yml' located in the current directory.

    Returns:
        dict: A dictionary containing parameters loaded from the YAML file.

    Raises:
        FileNotFoundError: If 'default_params.yml' does not exist in the current directory.
    """
    with open("./default_params.yml", "r") as file:
        return yaml.safe_load(file)


def load_pickle(path):
    """
    Load a pickle file from the specified path.

    Parameters:
        path (str): The path to the pickle file to be loaded.

    Returns:
        object: The object loaded from the pickle file.

    Raises:
        Exception: If the specified path does not exist.
    """
    if os.path.exists(path):
        return joblib.load(filename=path)
    else:
        raise Exception("Path is not found".capitalize())


def weight_init(m):
    """
    Initialize the weights of a PyTorch model following a normal distribution.

    Parameters:
        m (torch.nn.Module): The model or layer to initialize.

    Raises:
        ValueError: If the model is not defined (None).
    """
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
    """
    Initialize and return the specified device if available, otherwise default to CPU.

    Parameters:
        device (str): The name of the device ('cuda', 'mps', or 'cpu'). Default is 'mps'.

    Returns:
        torch.device: The initialized PyTorch device.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def ignore_warnings():
    """
    Suppress all warnings.
    """
    warnings.filterwarnings("ignore")


def saved_config(config_file=None, filename=None):
    """
    Save a configuration dictionary to a YAML file.

    Parameters:
        config_file (dict): The configuration dictionary to save.
        filename (str): The path to the file where the configuration will be saved, excluding the '.yml' extension.

    Raises:
        ValueError: If either 'config_file' or 'filename' is not provided.
    """
    if config_file is not None and filename is not None:
        with open(os.path.join(filename, ".yml"), "w") as file:
            yaml.safe_dump(config_file, file)
    else:
        raise ValueError("Define the arguments properly".capitalize())


def clean(activate=True):
    """
    Clean up checkpoints and test image directories by deleting their contents.

    Parameters:
        activate (bool): Whether to perform the cleanup. Default is True.

    Raises:
        ValueError: If 'activate' is not a boolean value.
    """
    if activate == True:
        if os.path.exists(TRAIN_CHECKPOINTS) and os.path.exists(LAST_CHECKPOINTS):
            for file in os.listdir(TRAIN_CHECKPOINTS):
                os.remove(os.path.join(TRAIN_CHECKPOINTS, file))

            for file in os.listdir(LAST_CHECKPOINTS):
                os.remove(os.path.join(LAST_CHECKPOINTS, file))

        if os.path.join(TEST_IMAGES):
            for file in os.listdir(TEST_IMAGES):
                os.remove(os.path.join(TEST_IMAGES, file))
    else:
        raise ValueError("Define the arguments properly".capitalize())
