from typing import List, Union

from torch import Tensor
import torch
from moonwatcher.utils.data import DataType
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.utils.data_storage import _prediction_name


class Annotation:
    def __init__(self, datapoint_number):
        self.datapoint_number = datapoint_number


class BoundingBoxes(Annotation):
    def __init__(
        self,
        datapoint_id: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
    ):
        """
        Initializes a BoundingBoxes object
        :param datapoint_id: The unique identifier for the data point.
        :param boxes_xyxy: A tensor of shape (num_boxes, 4) representing the bounding box coordinates.
        :param labels: An integer tensor of shape (num_boxes) representing labels for each bounding box.
        :return:
        """
        if not isinstance(boxes_xyxy, Tensor):
            raise TypeError("bounding boxes must be a Tensor of shape (num_boxes, 4)")
        if not isinstance(labels, Tensor):
            raise TypeError("labels must be an int Tensor of shape (num_boxes)")

        super().__init__(datapoint_id)
        self.boxes_xyxy = boxes_xyxy
        self.labels = labels

    def to_dict(self):
        return {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }


class PredictedBoundingBoxes(BoundingBoxes):
    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
        scores: Tensor,
    ):
        """
        Initializes a PredictedBoundingBoxes object
        :param datapoint_number: The unique identifier for the data point.
        :param boxes_xyxy: A tensor of shape (num_boxes, 4) representing bounding box coordinates.
        :param labels: An integer tensor of shape (num_boxes) representing labels for each bounding box.
        :param scores: A float tensor of shape (num_boxes) representing the confidence score for each bounding box.
        :return:
        """
        if not isinstance(scores, Tensor):
            raise TypeError("scores must be a float Tensor of shape (num_boxes)")

        super().__init__(datapoint_number, boxes_xyxy, labels)
        self.scores = scores

    def to_dict(self):
        return {
            "boxes": self.boxes_xyxy,
            "scores": self.scores,
            "labels": self.labels,
        }


class Labels(Annotation):
    def __init__(self, datapoint_number: int, labels: Tensor):
        """
        Initialize a Labels object
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor of shape (1) representing the label.
        :return:
        """
        # TODO Check if torchmetrics accepts labels both as torch.tensor([1]) and torch.tensor(1)
        if (
            not isinstance(labels, Tensor)
            or (labels.shape != (1,) and labels.shape != ())
            or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise TypeError("labels must be a 1-dimensional int Tensor")

        super().__init__(datapoint_number)
        self.labels = labels


class PredictedLabels(Labels):
    def __init__(self, datapoint_number: int, labels: Tensor, scores: Tensor):
        """
        Initialize a Labels object
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor of shape (1) representing the label.
        :param scores: A float tensor of shape (num_classes) representing the confidence scores for each class.
        :return:
        """
        # TODO Check if torchmetrics accepts labels both as torch.tensor([1]) and torch.tensor(1)
        if (
            not isinstance(scores, Tensor)
            or (scores.shape != (1,) and scores.shape != ())
            or scores.dtype not in (torch.float16, torch.float32, torch.float64)
        ):
            raise TypeError("scores must be a 1-dimensional float Tensor")

        super().__init__(datapoint_number, labels)
        self.scores = scores


class Annotations:
    def __init__(self, annotations: List[Annotation] = None):
        self.annotations = [] if annotations is None else annotations
        self.datapoint_number_to_annotation_index = {}
        for annotation_index, annotation in enumerate(self.annotations):
            self.datapoint_number_to_annotation_index[
                annotation.datapoint_number
            ] = annotation_index

    def add(self, annotation: Annotation):
        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[annotation.datapoint_number] = (
            len(self.annotations) - 1
        )

    def get(self, datapoint_number):
        return self.annotations[
            self.datapoint_number_to_annotation_index[datapoint_number]
        ]

    def get_datapoint_ids(self):
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number):
        return self.get(datapoint_number=datapoint_number)

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)


class Predictions(Annotations, MoonwatcherObject):
    def __init__(
        self,
        dataset,
        model,
        predictions: List[
            Union[PredictedBoundingBoxes, BoundingBoxes, PredictedLabels, Labels]
        ] = None,
    ):
        super().__init__(annotations=predictions)
        name = _prediction_name(model_name=model.name, dataset_name=dataset.name)

        MoonwatcherObject.__init__(self, name=name, datatype=DataType.PREDICTIONS)


class GroundTruths(Annotations, MoonwatcherObject):
    def __init__(
        self, dataset, groundtruths: List[Union[BoundingBoxes, Labels]] = None
    ):
        super().__init__(annotations=groundtruths)
        MoonwatcherObject.__init__(
            self, name=dataset.name, datatype=DataType.GROUNDTRUTHS
        )
