class ExperimentConfig:
    def __init__(self, d: dict):
        self.logger = d.get("logger", "neptune")
        self.projectName = d.get("projectName", None)
        self.experimentName = d.get("experimentName", None)
        self.tags = d.get("tags", [])

        # Resume an existing experiment. (e.g. 'SAN-45')
        self.experiment_id = d.get("experimentId", None)
