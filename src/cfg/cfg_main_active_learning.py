class ActiveLearn:
    def __init__(self, d: dict):
        self.init_size = d.get("init_size", 50)
        self.step_size = d.get("step_size", 50)
        self.mode = d.get("mode", 0)
        self.rerank_step = d.get("rerank_step", True)
        self.pixel_mode = d.get("pixel_mode", 0)
        self.pixel_percent = d.get("pixel_percent", 0.5)
        self.sp_mode = d.get("sp_mode", 0)
        self.sp_param = d.get("sp_param", 7000)
        self.num_cluster = d.get("num_cluster", 2)
        self.regen_mask = d.get("regen_mask", False)
        self.use_certain = d.get("use_certain", False)
        self.use_ignore = d.get("use_ignore", False)
