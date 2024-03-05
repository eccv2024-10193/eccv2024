from enum import Enum


class AnnotationType(Enum):
    CLASSIFICATION = "classification"
    SEGMANTIC = "segmantic"
    KEYPOINT = "keypoint"


class DatasetSubset(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class MetadataKey:
    IS_ALL_WEED_SM = "is_all_weed_sm"
    IS_ALL_FG_SM = "is_all_fg_sm"