import torch
import pytest
import numpy as np

from moonwatcher.utils.data import Task
from moonwatcher.inference.inference import inference
from moonwatcher.metric import calculate_metric_internal
from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.data_storage import (
    load_groundtruths,
    load_predictions,
    do_predictions_exist,
)


class MockModel:
    def __init__(self, name, task, device):
        self.name = name
        self.task = task
        self.device = device


class MockDataset(MoonwatcherDataset):
    def __init__(self, name):
        self.name = name
        self.dataset = []


class MockDetection:
    def __init__(self, boxes, labels, scores=None):
        self.boxes = torch.tensor(boxes)
        self.labels = torch.tensor(labels)
        self.scores = torch.tensor(scores) if scores else None

    def to_dict(self):
        result = {"boxes": self.boxes, "labels": self.labels}
        if self.scores is not None:
            result["scores"] = self.scores
        return result


def mock_do_predictions_exist(dataset_name, model_name):
    return True


def mock_load_groundtruths(dataset_name):
    return [
        MockDetection([[50, 50, 150, 150]], [1]),
        MockDetection([[30, 30, 120, 120]], [0]),
        MockDetection([[10, 10, 100, 100]], [1]),
        MockDetection([[40, 40, 140, 140]], [1]),
    ]


def mock_load_predictions(dataset_name, model_name):
    return [
        MockDetection([[50, 50, 150, 150]], [1], [0.9]),
        MockDetection([[35, 35, 115, 115]], [0], [0.8]),
        MockDetection([[15, 15, 105, 105]], [1], [0.75]),
        MockDetection([[45, 45, 145, 145]], [1], [0.85]),
    ]


def mock_inference(model, dataset, device):
    pass


@pytest.fixture(autouse=True)
def patch_functions(monkeypatch):
    monkeypatch.setattr(
        "moonwatcher.utils.data_storage.do_predictions_exist", mock_do_predictions_exist
    )
    monkeypatch.setattr(
        "moonwatcher.utils.data_storage.load_groundtruths", mock_load_groundtruths
    )
    monkeypatch.setattr(
        "moonwatcher.utils.data_storage.load_predictions", mock_load_predictions
    )
    monkeypatch.setattr("moonwatcher.inference.inference", mock_inference)


def load_data_for_testing():
    model = MockModel(name="mock_model", task=Task.DETECTION.value, device="cpu")
    dataset = MockDataset(name="mock_dataset")
    relevant_ids = list(range(4))
    groundtruths_loaded = mock_load_groundtruths(dataset.name)
    predictions_loaded = mock_load_predictions(dataset.name, model.name)
    return model, relevant_ids, dataset, groundtruths_loaded, predictions_loaded


def test_calculate_iou():
    (
        model,
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
    ) = load_data_for_testing()
    result = calculate_metric_internal(
        model,
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
        "IntersectionOverUnion",
    )
    assert result > 0.75, f"Expected IoU to be greater than 0.75 but got {result}"


def test_calculate_map():
    (
        model,
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
    ) = load_data_for_testing()
    result = calculate_metric_internal(
        model,
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
        "mAP",
    )
    assert result > 0.7, f"Expected mAP to be greater than 0.7 but got {result}"


if __name__ == "__main__":
    pytest.main()
