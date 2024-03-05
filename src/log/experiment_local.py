import json
import os
from pathlib import Path

import PIL.Image
from cfg.cfg_main import CFG, PCFG

from log.experiment_base import ExperimentBase


class ExperimentLocal(ExperimentBase):
    def __init__(self):
        self.exp_dir = CFG.experimentDir()
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)

        # Record CFG into a file.
        cfg_path = os.path.join(self.exp_dir, "cfg.json")
        cfg_dict = CFG.to_dict()
        with open(cfg_path, "w") as file:
            json.dump(cfg_dict, indent=2, fp=file)

        self.log_dir = CFG.logDir()
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.cur_im_idx = 0

    def log_metric(self, log_name, value):
        log_path = os.path.join(self.log_dir, f"{log_name}.txt")
        with open(log_path, "a") as file:
            file.write(f"{value}\n")

    def log_image(self, log_name: str, img_1: PIL.Image, description=None):
        if description is None:
            img_filename = f"{log_name}_{self.cur_im_idx}.png"
            self.cur_im_idx += 1
        else:
            img_filename = f"{log_name}_{description}.png"

        img_path = os.path.join(self.log_dir, img_filename)
        img_1.save(img_path)
