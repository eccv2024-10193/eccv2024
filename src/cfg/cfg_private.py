import argparse
import os
import sys
from typing import Union

import yaml

from cfg.cfg_private_logger import NeptuneConfig, WandBConfig

parser = argparse.ArgumentParser()
parser.add_argument("-p", type=str, dest="p", default="../cfg/private.yml")
args, _ = parser.parse_known_args()

# Singleton.
PCFG = None


class PrivateConfig:
    def __init__(self, yml_path: str):
        if not os.path.exists(yml_path):
            print(
                f"Private config YAML file '{yml_path}' does not exist. Make sure you "
                + "copied 'cfg/private_template.yml' to 'cfg/private.yml'."
            )
            sys.exit()

        ymlFile = open(yml_path)
        yml: dict = yaml.safe_load(ymlFile)

        """ Private settings """

        self.neptune = NeptuneConfig(yml.get("neptune", {}))
        self.wandb = WandBConfig(yml.get("wandb", {}))
        self.rootDataDir = yml.get("rootDataDir", None)

    def prepend_storage_dir_if_not_abspath(
        self, path: Union[str, None]
    ) -> Union[str, None]:
        """
        Given a path, prepend the storage directory to it if the path
        isn't an absolute path.
        """
        if path is None:
            return None

        if os.path.isabs(path):
            return path

        return os.path.join(self.rootDataDir, path)


PCFG = PrivateConfig(args.p)
