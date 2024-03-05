from cfg.cfg_private import PCFG


class Nassar2020Config:
    def __init__(self, d: dict):
        self.dir = PCFG.prepend_storage_dir_if_not_abspath(d.get("dir", ""))


class Cityscapes:
    def __init__(self, d: dict):
        self.dir = PCFG.prepend_storage_dir_if_not_abspath(d.get("dir", ""))


class DatasetsConfig:
    def __init__(self, d: dict):
        self.nassar = Nassar2020Config(d.get("nassar2020", {}))
        self.cityscapes = Cityscapes(d.get("cityscapes", {}))
