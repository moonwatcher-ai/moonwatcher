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
    def __init__(self, name, labels, label_to_name):
        self.name = name
        self.labels = labels
        self.label_to_name = label_to_name
        self.dataset = labels


class MockLabel:
    def __init__(self, label):
        self.labels = torch.tensor([label])


def mock_do_predictions_exist(dataset_name, model_name):
    return True


def mock_load_groundtruths(dataset_name):
    return [MockLabel(0), MockLabel(1), MockLabel(0), MockLabel(1)]


def mock_load_predictions(dataset_name, model_name):
    return [MockLabel(0), MockLabel(1), MockLabel(1), MockLabel(1)]


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
    model = MockModel(name="mock_model", task=Task.CLASSIFICATION.value, device="cpu")
    dataset = MockDataset(
        name="mock_dataset",
        labels=[0, 1, 0, 1],
        label_to_name={0: "class0", 1: "class1"},
    )
    relevant_ids = list(range(len(dataset.labels)))
    groundtruths_loaded = mock_load_groundtruths(dataset.name)
    predictions_loaded = mock_load_predictions(dataset.name, model.name)
    return model, relevant_ids, dataset, groundtruths_loaded, predictions_loaded


def test_calculate_accuracy():
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
        "Accuracy",
    )
    assert result == 0.75, f"Expected Accuracy to be 0.75 but got {result}"


def test_calculate_precision():
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
        "Precision",
    )
    assert result == 0.66667, f"Expected Precision to be 0.66667 but got {result}"


def test_calculate_recall():
    (
        model,
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
    ) = load_data_for_testing()
    result = calculate_metric_internal(
        model, relevant_ids, dataset, groundtruths_loaded, predictions_loaded, "Recall"
    )
    assert result == 1.0, f"Expected Recall to be 1.0 but got {result}"


def test_calculate_f1():
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
        "F1_Score",
    )
    assert result == 0.8, f"Expected F1_Score to be 0.8 but got {result}"


if __name__ == "__main__":
    pytest.main()
