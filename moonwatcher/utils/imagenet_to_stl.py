import json
from pathlib import Path


# Mapping from imagenet classes to slt10 classes
def load_mapping():
    cur_filepath = Path(__file__)
    with open(
        cur_filepath.parent.parent / "configs" / "imagenet_to_stl.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    stl10_classes = data["stl10_classes"]
    mapping = data["mapping"]

    def map_imagenet_to_stl10(imagenet_class_idx):
        return mapping.get(str(imagenet_class_idx), 9)

    return map_imagenet_to_stl10, stl10_classes


map_imagenet_to_stl10, stl10_classes = load_mapping()
