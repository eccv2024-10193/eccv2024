import neptune

from cfg.cfg_main import CFG
from cfg.cfg_private import PCFG
from log.experiment_base import ExperimentBase


class ExperimentNeptune(ExperimentBase):
    def __init__(self):
        # project_name = f"{PCFG.neptune.username}/{CFG.experiment.projectName}"
        project_name = f"{CFG.experiment.projectName}"

        print(f"Initializing Neptune project: {project_name}")

        neptune.init(
            project_qualified_name=project_name,
            api_token=PCFG.neptune.api_key,
        )

        neptune.create_experiment(
            name=CFG.experiment.experimentName,
            tags=CFG.experiment.tags,
            params=CFG.to_dict(),
        )

    def log_metric(self, log_name, value):
        neptune.log_metric(log_name, value)

    def log_image(self, log_name: str, img_1, description=None):
        neptune.log_image(log_name, img_1, description=description)

    def log_artifact(self, artifact):
        neptune.log_artifact(artifact)
