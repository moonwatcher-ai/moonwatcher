import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from moonwatcher.utils.data import DataType


class DataStorageManager:
    def __init__(self):
        load_dotenv()
        storage_folder = os.getenv("STORAGE_FOLDER")
        if storage_folder is None:
            storage_folder = "."

        self.storage_folder = Path(storage_folder) / "observations"
        self.storage_folder.mkdir(exist_ok=True)
        self.accepted_datatypes = [
            DataType.DATASET.value,
            DataType.SLICE.value,
            DataType.MODEL.value,
            DataType.CHECK.value,
            DataType.CHECKSUITE.value,
            DataType.PREDICTIONS.value,
            DataType.GROUNDTRUTHS.value,
        ]

    def get_filepath(self, datatype: DataType, name: str):
        if datatype.value not in self.accepted_datatypes:
            raise ValueError(
                f'Datatype: "{datatype.value}" is not an accepted datatype for data storage'
            )
        folder = self.storage_folder / datatype.value
        filepath = folder / (name + ".pt")
        filepath.parent.mkdir(exist_ok=True, parents=True)
        return filepath

    def store_file(self, file, datatype: DataType, name: str, overwrite=True):
        filepath = self.get_filepath(datatype=datatype, name=name)
        if not overwrite and filepath.exists():
            raise RuntimeError(
                f"Cannot store {datatype.value} {name}. It already exists."
            )
        torch.save(obj=file, f=filepath)
        return filepath

    def load_file(self, datatype: DataType, name: str):
        filepath = self.get_filepath(datatype=datatype, name=name)
        if not filepath.exists():
            raise RuntimeError(
                f"Cannot load {datatype.value} {name}. It does not exist."
            )
        return torch.load(f=filepath)

    def exists(self, datatype: DataType, name: str):
        filepath = self.get_filepath(datatype=datatype, name=name)
        return filepath.exists()


def exists(datatype: DataType, name: str):
    ds_manager = DataStorageManager()
    return ds_manager.exists(datatype=datatype, name=name)


def _prediction_name(model_name, dataset_name):
    return dataset_name + "_" + model_name


def _slice_name(dataset_name, name):
    return dataset_name + "_" + name


def do_predictions_exist(model_name, dataset_name):
    name = _prediction_name(model_name=model_name, dataset_name=dataset_name)
    return exists(datatype=DataType.PREDICTIONS, name=name)


def store_file(file, datatype: DataType, name: str, overwrite=False):
    ds_manager = DataStorageManager()
    return ds_manager.store_file(
        file=file, datatype=datatype, name=name, overwrite=overwrite
    )


def load(datatype: DataType, name: str):
    ds_manager = DataStorageManager()
    return ds_manager.load_file(datatype=datatype, name=name)


def load_model(name):
    return load(datatype=DataType.MODEL, name=name)


def load_dataset(name):
    dataset = load(datatype=DataType.DATASET, name=name)
    dataset.upload_if_not()
    return dataset


def load_slice(dataset_name, name):
    name = _slice_name(dataset_name=dataset_name, name=name)
    return load(datatype=DataType.SLICE, name=name)


def load_check(name):
    return load(datatype=DataType.CHECK, name=name)


def load_checksuite(name):
    return load(datatype=DataType.CHECKSUITE, name=name)


def load_predictions(dataset_name, model_name):
    return load(
        datatype=DataType.PREDICTIONS,
        name=_prediction_name(model_name=model_name, dataset_name=dataset_name),
    )


def load_groundtruths(dataset_name):
    return load(datatype=DataType.GROUNDTRUTHS, name=dataset_name)
