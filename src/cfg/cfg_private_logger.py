class NeptuneConfig:
    def __init__(self, d: dict):
        self.api_key = d.get("api_key")
        self.username = d.get("username")


class WandBConfig:
    def __init__(self, d: dict):
        self.username = d.get("username")