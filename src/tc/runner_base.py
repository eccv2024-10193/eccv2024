from cfg.cfg_main import CFG
from log.experiment_base import ExperimentBase
from log.experiment_local import ExperimentLocal
from log.experiment_neptune import ExperimentNeptune


class RunnerBase:
    def __init__(self):
        """
        Initialize a runner that can train or test a model.
        """
        self.exp: ExperimentBase = None

        if CFG.experiment.logger == "neptune":
            self.exp = self.get_neptune_logger()
        else:
            self.exp = ExperimentLocal()

    def get_neptune_logger(self):
        return ExperimentNeptune()

    def train(self):
        raise NotImplementedError("Not implemented.")

    def test(self):
        raise NotImplementedError("Not implemented.")

    def train_and_test(self):
        raise NotImplementedError("Not implemented.")
