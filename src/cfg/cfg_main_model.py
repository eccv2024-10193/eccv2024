class Unet:
    """
    Configuration relating to the Unet model.
    """

    def __init__(self, d: dict):
        self.numFilters = d.get("numFilters", 16)
        self.dropout = d.get("dropout", 0.2)
        self.dropcenter = d.get("dropcenter", None)


class Deeplab:
    """
    Configuration relating to the Unet model.
    """

    def __init__(self, d: dict):
        self.dropout = d.get("dropout", 0.2)


class ModelConfigs:
    def __init__(self, d: dict):
        self.Unet = Unet(d.get("Unet", {}))
        self.Deeplab = Deeplab(d.get("Deeplab", {}))
