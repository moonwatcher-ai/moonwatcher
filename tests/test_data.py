from moonwatcher.utils.data import OPERATOR_DICT, Task, DataType


def test_operator_dict():
    test_cases = [
        ("<", 3, 5, True),
        ("<", 5, 3, False),
        (">", 5, 3, True),
        (">", 3, 5, False),
        (">=", 5, 5, True),
        (">=", 4, 5, False),
        ("<=", 3, 3, True),
        ("<=", 5, 3, False),
        ("==", 5, 5, True),
        ("==", 5, 4, False),
        ("=", 5, 5, True),
        ("=", 5, 4, False),
        ("!=", 5, 4, True),
        ("!=", 5, 5, False),
    ]

    for operator, left, right, expected in test_cases:
        assert (
            OPERATOR_DICT[operator](left, right) == expected
        ), f"Failed for operator {operator} with values {left} and {right}"


def test_task_enum():
    assert Task.CLASSIFICATION.value == "classification"
    assert Task.DETECTION.value == "detection"


def test_data_type_enum():
    assert DataType.DATASET.value == "dataset"
    assert DataType.SLICE.value == "slice"
    assert DataType.MODEL.value == "model"
    assert DataType.CHECK.value == "check"
    assert DataType.CHECKSUITE.value == "checksuite"
    assert DataType.CHECK_REPORT.value == "check_report"
    assert DataType.CHECKSUITE_REPORT.value == "checksuite_report"
    assert DataType.PREDICTIONS.value == "predictions"
    assert DataType.GROUNDTRUTHS.value == "groundtruths"


def test_all_operators_present():
    expected_operators = {"<", ">", ">=", "<=", "==", "=", "!="}
    assert (
        set(OPERATOR_DICT.keys()) == expected_operators
    ), "Not all expected operators are present in the dictionary"
