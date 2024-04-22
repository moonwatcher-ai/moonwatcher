import pytest
import torch

from moonwatcher.annotations import (
    BoundingBoxes,
    PredictedBoundingBoxes,
    Labels,
    Annotations,
    PredictedLabels,
)


def tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)


# Tests for BoundingBoxes
@pytest.mark.parametrize(
    "datapoint_id, boxes, labels, valid",
    [
        (1, tensor([[1, 2, 3, 4]]), tensor([1]), True),
        (2, "not a tensor", tensor([1]), False),  # Invalid boxes
        (3, tensor([[1, 2, 3, 4]]), "not a tensor", False),  # Invalid labels
    ],
)
def test_bounding_boxes_initialization(datapoint_id, boxes, labels, valid):
    if valid:
        bbox = BoundingBoxes(datapoint_id, boxes, labels)
        assert bbox.datapoint_number == datapoint_id
        assert torch.equal(bbox.boxes_xyxy, boxes)
        assert torch.equal(bbox.labels, labels)
    else:
        with pytest.raises(TypeError):
            BoundingBoxes(datapoint_id, boxes, labels)


# Tests for PredictedBoundingBoxes
@pytest.mark.parametrize(
    "datapoint_id, boxes, labels, scores, valid",
    [
        (1, tensor([[1, 2, 3, 4]]), tensor([1]), tensor([0.99]), True),
        (2, tensor([[1, 2, 3, 4]]), tensor([1]), 0.99, False),  # Invalid scores
    ],
)
def test_predicted_bounding_boxes_initialization(
    datapoint_id, boxes, labels, scores, valid
):
    if valid:
        pred_bbox = PredictedBoundingBoxes(datapoint_id, boxes, labels, scores)
        assert pred_bbox.scores is not None
        assert torch.equal(pred_bbox.scores, scores)
    else:
        with pytest.raises(TypeError):
            PredictedBoundingBoxes(datapoint_id, boxes, labels, scores)


# Tests for Labels
@pytest.mark.parametrize(
    "datapoint_id, labels, expected_exception",
    [
        (1, tensor([1], dtype=torch.int32), None),  # Correct case
        (2, tensor([1.3]), TypeError),  # Non-integer labels
        (3, "not a tensor", TypeError),  # Non-tensor labels
        (4, tensor([1, 2], dtype=torch.int32), TypeError),  # Incorrect shape
    ],
)
def test_labels_initialization(datapoint_id, labels, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            Labels(datapoint_id, labels)
    else:
        label_obj = Labels(datapoint_id, labels)
        assert label_obj.labels.shape == (1,)
        assert label_obj.labels.dtype == torch.int32


# Tests for PredictedLabels
@pytest.mark.parametrize(
    "datapoint_id, labels, scores, expected_exception",
    [
        (1, tensor([1], dtype=torch.int32), tensor([0.9]), None),  # Correct case
        (
            2,
            tensor([1], dtype=torch.int32),
            "not a tensor",
            TypeError,
        ),  # Non-tensor scores
        (
            3,
            tensor([1], dtype=torch.int32),
            tensor([0.9, 0.1]),
            TypeError,
        ),  # Incorrect scores shape
        (4, "not a tensor", tensor([0.9]), TypeError),  # Non-tensor labels
        (
            5,
            tensor([1, 2], dtype=torch.int32),
            tensor([0.9]),
            TypeError,
        ),  # Incorrect labels shape
    ],
)
def test_predicted_labels_initialization(
    datapoint_id, labels, scores, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            PredictedLabels(datapoint_id, labels, scores)
    else:
        predicted_label_obj = PredictedLabels(datapoint_id, labels, scores)
        assert predicted_label_obj.labels.shape == (1,)
        assert predicted_label_obj.labels.dtype == torch.int32
        assert predicted_label_obj.scores.shape == (1,)
        assert predicted_label_obj.scores.dtype == torch.float32


# Test for Annotations Class
def test_annotations():
    bbox = BoundingBoxes(1, tensor([[0, 1, 2, 3]]), tensor([1]))
    annotations = Annotations([bbox])
    retrieved = annotations.get(1)
    assert retrieved == bbox
    assert len(annotations) == 1
