from moonwatcher.datapoint import Datapoint


def test_datapoint_initialization():
    datapoint = Datapoint(10, "http://fakeurl.com/image_10", {"brightness": 0.8})
    assert datapoint.number == 10
    assert datapoint.locator == "http://fakeurl.com/image_10"
    assert datapoint.metadata == {"brightness": 0.8}


def test_add_metadata():
    datapoint = Datapoint(10)
    datapoint.add_metadata("brightness", 0.8)
    assert datapoint.metadata == {"brightness": 0.8}


def test_get_metadata():
    datapoint = Datapoint(10, metadata={"brightness": 0.8})
    assert datapoint.get_metadata("brightness") == 0.8
    assert datapoint.get_metadata("nonexistent_key") is None
