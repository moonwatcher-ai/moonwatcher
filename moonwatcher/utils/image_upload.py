import json
from pathlib import Path
from typing import Union, Optional

from tqdm import tqdm
import boto3


def upload_image(
    s3_client,
    file_path: Union[Path, str],
    bucket_name: str,
    upload_folder: Optional[str] = None,
    region="eu-central-1",
):
    file_path = Path(file_path)

    upload_path = file_path.name
    if upload_folder is not None:
        upload_path = f"{upload_folder}/{upload_path}"

    s3_client.upload_file(
        file_path, bucket_name, upload_path, ExtraArgs={"ContentType": f"image/jpeg"}
    )

    url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{upload_path}"
    return url


def upload_images(
    image_folder: Union[Path, str],
    upload_folder: str,
    bucket_name: str,
    s3_client,
    urls_json_file_path: str = None,
    region: str = "eu-central-1",
):
    image_folder = Path(image_folder)

    image_paths = sorted([image_path for image_path in image_folder.iterdir()])

    urls = []
    print(
        f"Uploading images from {image_folder.as_posix()} to s3 bucket {bucket_name} in folder {upload_folder}"
    )
    for image_path in tqdm(image_paths, desc="Uploading images"):
        if image_path.name == ".DS_Store":
            continue
        url = upload_image(
            s3_client=s3_client,
            file_path=image_path,
            bucket_name=bucket_name,
            upload_folder=upload_folder,
            region=region,
        )
        urls.append(url)

    if urls_json_file_path is not None:
        with open(urls_json_file_path, "w", encoding="utf-8") as f:
            json.dump(obj=urls, fp=f, indent=4)


if __name__ == "__main__":
    upload_images(
        "../../data/caltech",
        "demo/classification/caltech101",
        "moonwatcher-webapp-dev",
        boto3.client("s3"),
        "../configs/caltech_locators.json",
    )
