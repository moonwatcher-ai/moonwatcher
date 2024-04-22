import boto3
import torchvision.datasets

from moonwatcher.utils.image_upload import upload_images


image_folder = "../data/COCO/val2017/"
annotations_file = "../data/COCO/annotations/instances_val2017.json"

dataset = torchvision.datasets.CocoDetection(
    root=image_folder,
    annFile=annotations_file,
)

s3 = boto3.client("s3")


upload_images(
    image_folder=image_folder,
    upload_folder="demo/images",
    s3_client=s3,
    bucket_name="moonwatcher-webapp-dev",
    urls_json_file_path="coco_upload.json",
)
