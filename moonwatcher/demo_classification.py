import json
import random
from pathlib import Path

import torch
import torchvision.datasets
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Subset
from torchvision.models import ResNet50_Weights

from moonwatcher.model.model import MoonwatcherModel, ModelOutputInputTransformation
from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.data_storage import load_model, load_dataset
from moonwatcher.utils.data import Task
from moonwatcher.check import automated_checking
from moonwatcher.utils.imagenet_to_stl import map_imagenet_to_stl10, stl10_classes

# TODO 1) Choose a Model
_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
_model = _model.eval()


# TODO 2) Choose a Dataset
image_folder = "."
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
_dataset = torchvision.datasets.STL10(
    root=image_folder,
    transform=transform,
    split="test",
    download=True,
)


# TODO 3) Write transformations for model
class ModelOutputInputTransformation(ModelOutputInputTransformation):
    def __init__(self):
        pass

    def transform_input(self, image):
        return image.unsqueeze(0)

    def transform_output(self, outputs):
        outputs = outputs.detach()
        imagenet_class_idx = torch.argmax(outputs, dim=1).item()
        # We are using a resnet trained on imagenet to make predictions on stl10
        # Therefore a remapping is necessary to map the imagenet classes to the stl10 classes
        stl10_class_idx = map_imagenet_to_stl10(imagenet_class_idx)
        return torch.tensor([stl10_class_idx])


# TODO 4) Write transformations for dataset
def dataset_output_transform(data):
    x, label = data
    return x, torch.tensor(label)


# TODO 5) Create Moonwatcher Dataset and Model
model_name = f"ResNet50"
dataset_name = f"STL10"

try:
    mw_model = load_model(model_name)
except:
    mw_model = MoonwatcherModel(
        model=_model,
        name=model_name,
        task=Task.CLASSIFICATION.value,
        output_input_transform=ModelOutputInputTransformation(),
        device="cpu",
    )

try:
    mw_dataset = load_dataset(dataset_name)
except:
    # Mapping from numerical labels to strings
    label_to_name = dict(enumerate(stl10_classes))

    # Select subset of dataset
    n_samples = 100
    random.seed(42)
    random_indices = random.sample(range(len(_dataset)), n_samples)
    _dataset = Subset(_dataset, random_indices)

    mw_dataset = MoonwatcherDataset(
        dataset=_dataset,
        name=dataset_name,
        task=Task.CLASSIFICATION.value,
        output_transform=dataset_output_transform,
        label_to_name=label_to_name,
    )


# TODO 6) Automated Checking
automated_checking(
    mw_dataset=mw_dataset,
    mw_model=mw_model,
)
