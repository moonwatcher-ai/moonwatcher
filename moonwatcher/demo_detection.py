import os
import json
from pathlib import Path

import torch
import numpy as np
import torchvision.datasets
from torch.utils.data import Subset
from torchvision.transforms import functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection

from moonwatcher.utils.data import Task
from moonwatcher.check import automated_checking
from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.bbox_utils import box_xywh_abs_to_xyxy_abs
from moonwatcher.utils.data_storage import load_model, load_dataset
from moonwatcher.model.model import MoonwatcherModel, ModelOutputInputTransformation


# TODO 1) Choose a Model
_model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm"
)
_model = _model.eval()


# TODO 2) Choose a Dataset
cur_filepath = Path(__file__)
coco_path = Path("coco")
coco_sh_path = cur_filepath.parent / "coco.sh"
if not coco_path.exists() or not any(coco_path.iterdir()):
    os.system(f"sh {coco_sh_path}")

image_folder = "coco/images/val2017/"
annotations_file = "coco/annotations/instances_val2017.json"

_dataset = torchvision.datasets.CocoDetection(
    root=image_folder,
    annFile=annotations_file,
)


# TODO 3) Write transformations for model
class ModelOutputInputTransformation(ModelOutputInputTransformation):
    def __init__(self):
        self.target_sizes = None
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )

    def transform_input(self, images):
        self.target_sizes = torch.tensor([images.size[::-1]])
        inputs = self.processor(images=images, return_tensors="pt")
        return [], inputs

    def transform_output(self, outputs):
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=self.target_sizes, threshold=0.5
        )[0]
        scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
        return boxes, labels, scores


# TODO 5) Write transformations for dataset
def dataset_output_transform(data):
    pil_image, annotation_list = data
    boxes = []
    labels = []
    for annotation in annotation_list:
        box_xywh_abs = annotation["bbox"]
        box_xyxy_abs = box_xywh_abs_to_xyxy_abs(box_xywh_abs)
        label = annotation["category_id"]
        boxes.append(box_xyxy_abs)
        labels.append(label)
    x = F.to_tensor(pil_image)
    boxes = torch.tensor(boxes)
    labels = torch.tensor(labels, dtype=torch.int64)
    return x, boxes, labels


# TODO 4) Create Moonwatcher Dataset and Model
appendix = f"{np.random.randint(0,100)}"
model_name = f"facebook-detr-resnet50"
dataset_name = f"COCO_val2017_subset"

try:
    mw_model = load_model(model_name)
except:
    mw_model = MoonwatcherModel(
        model=_model,
        name=model_name,
        task=Task.DETECTION.value,
        output_input_transform=ModelOutputInputTransformation(),
        device="cpu",
    )

try:
    mw_dataset = load_dataset(dataset_name)
except:
    # Mapping from numerical labels to strings
    label_to_name = {key: _dataset.coco.cats[key]["name"] for key in _dataset.coco.cats}

    # Select Subset of Dataset
    n_samples = 100
    _dataset = Subset(_dataset, [i for i in range(n_samples)])

    mw_dataset = MoonwatcherDataset(
        dataset=_dataset,
        name=dataset_name,
        task=Task.DETECTION.value,
        output_transform=dataset_output_transform,
        label_to_name=label_to_name,
    )


# TODO 6) Automated Checking
automated_checking(
    mw_dataset,
    mw_model,
)
