import argparse
import os
import sys
from typing import Optional

import yaml

from cfg.cfg_main_active_learning import ActiveLearn
from cfg.cfg_main_experiment import ExperimentConfig
from cfg.cfg_main_model import ModelConfigs
from cfg.cfg_main_dataset import DatasetsConfig
from ds.ds_def import DatasetSubset
from util import serialization

parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str, dest="f", default="../cfg/config.yml")
args, _ = parser.parse_known_args()

# Singleton.
CFG = None

# Forward import.
# pylint: disable=unused-import, disable=wrong-import-position
from cfg.cfg_private import PCFG


class MainConfig:
    """
    Basic configuration for a run. The configuration can be initialized from a .yml file,
    and then used as a singleton that's accessible from anywhere.

    Having a singleton configuration reduces a lot of config passing across functions.
    """

    def __init__(self, yml_path: str):
        if not os.path.exists(yml_path):
            print(
                f"Config YAML file '{yml_path}' does not exist. Make sure you "
                + "copied 'cfg/config_template.yml' to 'cfg/config.yml'."
            )
            sys.exit()

        self.yml_path = yml_path
        ymlFile = open(yml_path)
        yml: dict = yaml.safe_load(ymlFile)

        """ Runner """

        self.runner = yml.get("runner", "th_mnist")
        self.is_debug_run = yml.get("is_debug_run", False)

        """ Initial config """

        self.num_gpus = yml.get("num_gpus", 1)

        """ Pathway """

        # "train", "test"
        self.mode = DatasetSubset(yml.get("mode", "train"))

        self.is_test_on_val_dataset: bool = yml.get("is_test_on_val_dataset", False)
        self.test_vis: bool = yml.get("test_vis", False)

        """ Input dataset """

        self.datasets = DatasetsConfig(yml.get("datasets", {}))

        """ Active learning"""
        self.activeLearn = ActiveLearn(yml.get("active_learn", {}))

        """ Common training settings """

        self.numEpochs = yml.get("numEpochs", 1)
        self.batchSize = yml.get("batchSize", 1)
        self.lr = yml.get("lr", 0.0002)

        """ Models """

        self.models = ModelConfigs(yml.get("models", {}))

        """ Local Performance """

        self.numWorkers = yml.get("numWorkers", 0)

        """ Experiment Tracking (e.g. Neptune) """

        self.experiment = ExperimentConfig(yml.get("experiment", {}))

    def experimentDir(self, project_name: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Get the directory for the current experiment. Output relating the experiment
        can be placed in the directory. For logs and checkpoints, in particular,
        use convenience methods logDir() and checkpointDir().
        """
        if project_name is None:
            project_name = self.experiment.projectName

        if experiment_name is None:
            experiment_name = self.experiment.experimentName

        dir_path = os.path.join("projects", project_name, experiment_name)

        if self.is_debug_run:
            dir_path = os.path.join(dir_path, "debug_run")

        dir_path = PCFG.prepend_storage_dir_if_not_abspath(dir_path)
        return dir_path

    def logDir(self):
        """
        Get the log directory for the current experiment.
        """
        return os.path.join(self.experimentDir(), f"{self.mode.value}_logs")

    def checkpointDir(self, project_name: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Get the checkpoint directory for the current experiment.
        """
        return os.path.join(
            self.experimentDir(project_name=project_name, experiment_name=experiment_name),
            "checkpoints",
        )

    def testDir(self):
        """
        Get the directory for any testcase output.
        """
        return PCFG.prepend_storage_dir_if_not_abspath("test")

    def cache_dir(self):
        """
        Get the directory where cached items (e.g. pre-resized) images can be stored in.
        """
        return PCFG.prepend_storage_dir_if_not_abspath("cache")

    def tile_dir(self):
        """
        Get the directory where cached items (e.g. pre-resized) images can be stored in.
        """
        return PCFG.prepend_storage_dir_if_not_abspath("tile")

    def to_dict(self) -> dict:
        """
        Convert this config object into a dictionary. Useful for serialization.
        """
        return serialization.object_to_dict(self)


CFG = MainConfig(args.f)
