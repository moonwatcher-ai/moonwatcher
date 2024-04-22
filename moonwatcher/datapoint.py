class Datapoint:
    def __init__(self, number, locator: str = None, metadata: dict = None):
        self.number = number
        self.locator = locator
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def add_metadata(self, key: str, value: any):
        self.metadata[key] = value

    def get_metadata(self, key):
        return self.metadata.get(key, None)
