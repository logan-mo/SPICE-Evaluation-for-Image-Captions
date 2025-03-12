import os
import boto3
import random
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client("s3")

bucket_name = "semantic-search-es8"
image_folder = "photos_for_semantic_search_poc/"
alt_text_folder = "celeb_alt_text_for_photos_semantic_search"
metadata_folder = "photos_metadata_for_semantic_search_poc/"

bucket_name = os.getenv("bucket_name")
image_folder = os.getenv("image_folder")
alt_text_folder = os.getenv("alt_text_folder")
metadata_folder = os.getenv("metadata_folder")


def download_image_from_s3(bucket_name, image_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    return response["Body"].read()  # Return raw image bytes


def get_random_image(bucket_name=bucket_name, folder=image_folder):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    image_keys = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return random.choice(image_keys) if image_keys else None
