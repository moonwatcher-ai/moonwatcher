import torch

from moonwatcher.utils.helpers import get_current_timestamp, convert_to_list


def test_get_current_timestamp():
    timestamp = get_current_timestamp()

    assert isinstance(timestamp, str)
    assert len(timestamp) == 32
    assert timestamp[4] == "-"
    assert timestamp[7] == "-"
    assert timestamp[10] == "T"
    assert timestamp[13] == ":"
    assert timestamp[16] == ":"
    assert timestamp[19] == "."
    assert timestamp[26] == "+"
    assert timestamp[-6:] == "+02:00"


def test_convert_to_list():
    items = torch.tensor([7, 8, 9])
    assert convert_to_list(items) == items.tolist()

    items = torch.tensor([7.3, 8.0, 9])
    assert convert_to_list(items) == items.tolist()

    items = torch.tensor(7)
    assert convert_to_list(items) == [items]

    items = torch.tensor(2.3)
    assert convert_to_list(items) == [items]
