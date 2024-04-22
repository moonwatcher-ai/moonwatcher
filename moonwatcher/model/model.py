from typing import Dict, Any
from abc import ABC, abstractmethod

from torch.nn import Module

from moonwatcher.utils.data import DataType
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.utils.helpers import get_current_timestamp
from moonwatcher.utils.api_connector import upload_if_possible


class ModelOutputInputTransformation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform_input(self, inputs):
        """
        Transform input data into a format that can be directly passed to the model.

        :param inputs: An image from your specified dataset.
        :return:
            The transformed input data, formatted to be passed directly to the model.
            The output can be structured either as positional arguments (*args),
            or as both positional and keyword arguments (*args and **kwargs).

        Internally, the arguments are directly passed to the model either as model(*args) or model(*args, **kwargs).
        """
        pass

    @abstractmethod
    def transform_output(self, outputs):
        """
        Transform the output of the model into the required format.

        :param outputs: The output from the model.

        :return:
            A tuple containing the model outputs formatted as required:
            - Classification:
                labels (torch.Tensor): A 1-dimensional integer tensor of shape (1) representing the label.
                scores (optional, torch.Tensor): A float tensor of shape (num_classes) representing the confidence scores for each class.

            - Detection:
                boxes_xyxy (torch.Tensor): A tensor of shape (num_boxes, 4) representing bounding box coordinates.
                labels (torch.Tensor): An integer tensor of shape (num_boxes) representing labels for each bounding box.
                scores (optional, torch.Tensor): A float tensor of shape (num_boxes) representing the confidence score for each bounding box.
        """
        pass


class MoonwatcherModel(MoonwatcherObject, Module):
    def __init__(
        self,
        model: Module,
        name: str,
        task: str,
        output_input_transform: ModelOutputInputTransformation,
        device: str,
        metadata: Dict[str, Any] = None,
        description: str = None,
    ):
        """
        Creates a moonwatcher model wrapper around an existing model that can be used with the moonwatcher framework

        :param model: the model to be wrapped
        :param name: the name you want to give this model
        :param task: either classification or detection
        :param output_input_transform: see ModelOutputInputTransformation class, formatting input output for moonwatcher
        :param device: only cpu works for now
        :param metadata: dictionary of tags for the model, can be ignored
        :param description: description of the model, can be ignored
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.MODEL)

        Module.__init__(self)

        self.task = task
        self.model = model
        self.metadata = metadata or {}
        self.timestamp = get_current_timestamp()
        self.description = description
        self.output_input_transform = output_input_transform
        self.device = device
        self.store()

    def _upload(self):
        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "task": self.task,
        }
        return upload_if_possible(datatype=DataType.MODEL.value, data=data)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
