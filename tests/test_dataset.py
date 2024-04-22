import uuid

import torch
import pytest
import numpy as np
from torch.utils.data import Dataset

from moonwatcher.metric import get_original_indices
from moonwatcher.dataset.dataset import MoonwatcherDataset, Slice


class MockDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        dummy_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        return dummy_image, idx % 10


@pytest.fixture
def simple_dataset():
    return MockDataset()


def output_transform(x):
    return x[0], torch.tensor(x[1])


@pytest.fixture
def basic_moonwatcher_dataset(simple_dataset):
    unique_name = f"test_dataset_{uuid.uuid4()}"
    return MoonwatcherDataset(
        dataset=simple_dataset,
        name=unique_name,
        task="classification",
        output_transform=output_transform,
        label_to_name={i: f"class_{i}" for i in range(10)},
        locators=["http://fakeurl.com/image_{}".format(i) for i in range(100)],
    )


def test_initialization(basic_moonwatcher_dataset):
    assert basic_moonwatcher_dataset.name.startswith("test_dataset_")
    assert len(basic_moonwatcher_dataset) == 100
    assert isinstance(basic_moonwatcher_dataset, MoonwatcherDataset)


def test_data_retrieval(basic_moonwatcher_dataset):
    data_point = basic_moonwatcher_dataset.get_datapoint(10)
    assert data_point is not None
    assert data_point.number == 10


def test_metadata_addition(basic_moonwatcher_dataset):
    basic_moonwatcher_dataset.add_predefined_metadata(
        predefined_metadata_key="brightness",
    )
    assert "brightness" in basic_moonwatcher_dataset.datapoints[0].metadata


def test_slicing_by_threshold(basic_moonwatcher_dataset):
    basic_moonwatcher_dataset.add_predefined_metadata(
        predefined_metadata_key="brightness",
    )
    slice_dataset = basic_moonwatcher_dataset.slice_by_threshold("brightness", ">", 190)
    for idx in slice_dataset.indices:
        brightness = basic_moonwatcher_dataset.datapoints[idx].metadata["brightness"]
        assert brightness > 190, f"Expected brightness > 0.1 but got {brightness}"


def test_slicing_by_percentile(basic_moonwatcher_dataset):
    basic_moonwatcher_dataset.add_predefined_metadata(
        predefined_metadata_key="contrast",
    )
    slice_dataset = basic_moonwatcher_dataset.slice_by_percentile("contrast", ">", 90)
    assert len(slice_dataset) < len(basic_moonwatcher_dataset)


def test_slicing_by_class(basic_moonwatcher_dataset):
    basic_moonwatcher_dataset.add_metadata_custom(
        metadata_key="class_type",
        metadata_func=lambda x: 0 if np.random.rand() < 0.5 else 1,
    )
    slices = basic_moonwatcher_dataset.slice_by_class("class_type")
    assert len(slices) == 2


def test_get_original_indices(basic_moonwatcher_dataset):
    slice_indices = [i for i in range(10, 20)]
    mock_slice = Slice(
        basic_moonwatcher_dataset, "Slice1", slice_indices, basic_moonwatcher_dataset
    )

    indices = get_original_indices(mock_slice)
    assert indices == list(range(10, 20))

    indices = get_original_indices(basic_moonwatcher_dataset)
    assert indices == list(range(100))

    with pytest.raises(TypeError):
        get_original_indices("invalid_input")


def test_get_original_indices_nested(basic_moonwatcher_dataset):
    # First-level slice
    slice_indices = [i for i in range(10, 20)]
    mock_slice = Slice(
        basic_moonwatcher_dataset,
        f"Slice1_{uuid.uuid4()}",
        slice_indices,
        basic_moonwatcher_dataset,
    )

    indices = get_original_indices(mock_slice)
    assert indices == list(range(10, 20))

    # Second-level slice (slice of a slice)
    nested_slice_indices = [i for i in range(5, 10)]
    nested_mock_slice = Slice(
        mock_slice,
        f"NestedSlice_{uuid.uuid4()}",
        nested_slice_indices,
        basic_moonwatcher_dataset,
    )

    nested_indices = get_original_indices(nested_mock_slice)
    assert nested_indices == list(range(15, 20))

    # Full dataset indices
    indices = get_original_indices(basic_moonwatcher_dataset)
    assert indices == list(range(100))

    # Invalid input
    with pytest.raises(TypeError):
        get_original_indices("invalid_input")
