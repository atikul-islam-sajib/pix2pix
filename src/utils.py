import yaml


def params():
    with open("./default_params.yml", "r") as file:
        config = yaml.safe_load(file)
