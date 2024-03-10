import sys
import os
import yaml
import joblib

sys.path.append("src/")


def params():
    with open("./default_params.yml", "r") as file:
        return yaml.safe_load(file)


def load_pickle(path):
    if os.path.exists(path):
        return joblib.load(filename=path)
    else:
        raise Exception("Path is not found".capitalize())
