
# import packages
import os
import ruamel.yaml
from pathlib import Path

PROJECT_NAME = "soccer_analytics"
CONFIG_NAME = "config.yml"
ALL_LEAGUES = ["germany", "italy", "england", "spain", "france"]


def _get_config_file():
    """
    Helper function to retrieve the name of the config-file
    :return: String with the config file name
    """

    base_path = Path(__file__).parent
    file_path = (base_path / "../config.yml").resolve()

    with open(file_path, "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    return config


if __name__ == "__main__":

    hallo = _get_config_file()
    print(hallo)