import pytest

from moonwatcher.base.base import MoonwatcherObject, DataType

valid_names = [
    "exampleName",
    "example_name",
    "Example-Name",
    "example123",
    "123example",
    "name_123",
    "123_name",
    "example-name123",
    "name123-name",
    "abc_xyz-123",
    "data_set_1",
    "data-set-2",
    "Test_Name_3",
    "Test-Name-4",
    "dataset5",
    "dataSet6",
    "Data_Set-7",
    "Data-Set_8",
    "example_9_name",
    "example_10-name",
]

invalid_names = [
    "hello world",
    "data@set",
    "profile#1",
    "update%20",
    "name*star",
    "test(case)",
    "click&collect",
    "math+science",
    "value=truth",
    "key|value",
    "path/to/file",
    "user\\admin",
    "comma,separated",
    "semicolon;",
    "colon:colon",
    'quote"quote',
    "apostrophe's",
    "bracket[1]",
    "bracket{2}",
    "less<than",
]


def test_valid_name():
    for name in valid_names:
        obj = MoonwatcherObject(name, DataType.DATASET)
        assert obj.valid_name() == True

    for name in invalid_names:
        with pytest.raises(ValueError):
            MoonwatcherObject(name, DataType.DATASET)
