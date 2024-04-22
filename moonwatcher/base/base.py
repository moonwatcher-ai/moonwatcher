import re
from abc import abstractmethod

from moonwatcher.utils.data import DataType
from moonwatcher.utils.data_storage import store_file, exists, _slice_name


class MoonwatcherObject:
    def __init__(self, name, datatype: DataType):
        self.uploaded = False
        self.name = name
        self.datatype = datatype
        if not self.valid_name():
            raise ValueError(
                f"Invalid name '{self.name}' for {datatype.value}! Only use a-z, A-Z, 0-9, hyphen (-) or underscore (_)!"
            )
        if self.exists_already():
            raise ValueError(f"{datatype.value} with name {self.name} already exists!")
        pass

    @abstractmethod
    def _upload(self) -> bool:
        raise NotImplementedError

    def upload_if_not(self):
        if not self.uploaded:
            upload_successful = self._upload()
            self.uploaded = upload_successful

    def store(self, overwrite=True):
        name = self.name
        if self.datatype.value == DataType.SLICE.value:
            name = _slice_name(dataset_name=self.dataset_name, name=name)
        store_file(
            file=self,
            datatype=self.datatype,
            name=name,
            overwrite=overwrite,
        )

    def exists_already(self):
        name = self.name
        if self.datatype.value == DataType.SLICE.value:
            name = _slice_name(dataset_name=self.dataset_name, name=name)
        exists(datatype=self.datatype, name=name)

    def valid_name(self):
        """
        Checks if a name contains only the following characters:
        Alphanumeric (a-z, A-Z, 0-9), underscore (_), and hyphen (-).
        """
        pattern = r"^[a-zA-Z0-9\_\-]+$"
        return bool(re.match(pattern, self.name))


# TODO Might be used in some way in the future
class MoonwatcherWrapper:
    def __init__(self, object):
        self.wrapped_object = object

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_object, attr)

    def __setattr__(self, attr, value):
        if attr == "wrapped_object":
            super().__setattr__(attr, value)
        else:
            setattr(self.wrapped_object, attr, value)
