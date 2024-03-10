import yaml


def params():
    with open("./default_params.yml", "r") as file:
        return yaml.safe_load(file)
