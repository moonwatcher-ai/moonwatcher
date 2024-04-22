import operator as op
from enum import Enum


OPERATOR_DICT = {
    "<": op.lt,
    ">": op.gt,
    ">=": op.ge,
    "<=": op.le,
    "==": op.eq,
    "=": op.eq,
    "!=": op.ne,
}


class Task(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class DataType(Enum):
    DATASET = "dataset"
    SLICE = "slice"
    MODEL = "model"
    CHECK = "check"
    CHECKSUITE = "checksuite"
    CHECK_REPORT = "check_report"
    CHECKSUITE_REPORT = "checksuite_report"
    PREDICTIONS = "predictions"
    GROUNDTRUTHS = "groundtruths"
