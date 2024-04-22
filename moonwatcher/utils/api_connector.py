import os
import json
from typing import Dict, Union, List

import requests
from dotenv import load_dotenv

from moonwatcher.utils.data import DataType


def check_api_call_success(response):
    if not response.ok:
        raise RuntimeError(f"Upload API Call: {response.status_code} - {response.text}")


def is_api_key_and_endpoint_available():
    load_dotenv()
    base_url = os.getenv("API_ENDPOINT")
    api_key = os.getenv("API_KEY")
    if (base_url is None) or (api_key is None):
        return False
    return True



class ApiConnector:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("API_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        self.available = True
        if (self.base_url is None) or (self.api_key is None):
            self.available = False

        self.headers = {"X-API-KEY": self.api_key}

    def hello(self):
        r = requests.get(url=self.base_url + "hello", headers=self.headers)
        print(r)
        body = r.content.decode()
        print(body)
        pass

    def get_presigned_post(self, filename: str, **kwargs):
        headers = {key: self.headers[key] for key in self.headers}
        headers.update({k: kwargs[k] for k in kwargs})
        headers["filename"] = filename
        r = requests.get(url=self.base_url + "get_presigned_post", headers=headers)
        check_api_call_success(response=r)
        body = json.loads(r.content.decode())
        return body


class DataUploader:
    def __init__(self):
        self.api_connector = ApiConnector()
        self.upload_possible = self.api_connector.available

    def upload(self, datatype: str, upload_data: Union[Dict, List]):
        accepted_datatypes = [upload_datatype.value for upload_datatype in DataType]
        kwargs = {}
        if datatype not in accepted_datatypes:
            raise ValueError(
                f"Datatype {datatype} is not accepted for upload. Accepted datatypes are {accepted_datatypes}"
            )
        if datatype in [
            DataType.DATASET.value,
            DataType.MODEL.value,
            DataType.CHECK.value,
            DataType.CHECKSUITE.value,
        ]:
            name = upload_data["name"]
        elif datatype == DataType.SLICE.value:
            name = f"{upload_data['dataset_name']}_{upload_data['name']}"
            kwargs["dataset_name"] = upload_data["dataset_name"]
            kwargs["slice_name"] = upload_data["name"]
        elif datatype == DataType.CHECK_REPORT.value:
            name = f"{upload_data['model_name']}_{upload_data['check_name']}"
        elif datatype == DataType.CHECKSUITE_REPORT.value:
            name = f"{upload_data['model_name']}_{upload_data['checksuite_name']}"
        elif datatype == DataType.GROUNDTRUTHS.value:
            name = f"{upload_data[0]['dataset_name']}"
        elif datatype == DataType.PREDICTIONS.value:
            name = f"{upload_data[0]['dataset_name']}_{upload_data[0]['model_name']}"

        filename = f"{datatype}__{name}.json"
        response = self.api_connector.get_presigned_post(filename=filename, **kwargs)
        files = {"file": json.dumps(upload_data)}
        data = response["fields"]
        r = requests.post(response["url"], data=data, files=files)
        check_api_call_success(response=r)
        print(f'Uploaded {datatype} as file "{filename}"')


def upload_if_possible(datatype: str, data: Union[Dict, List]):
    uploader = DataUploader()
    if uploader.upload_possible:
        uploader.upload(datatype=datatype, upload_data=data)
        return True
    return False


if __name__ == "__main__":
    api_connector = ApiConnector()
    r = api_connector.hello()
