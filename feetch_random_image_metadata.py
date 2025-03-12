import base64
import boto3
import json
import os
import io
import random
import pandas as pd
from IPython.display import display, Image as IPImage

s3_client = boto3.client("s3")

bucket_name = os.getenv("bucket_name")
image_folder = os.getenv("image_folder")
alt_text_folder = os.getenv("alt_text_folder")
metadata_folder = os.getenv("metadata_folder")
csv_key = os.getenv("csv_key")


def get_random_image(bucket_name, folder):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    image_keys = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return random.choice(image_keys) if image_keys else None


def download_image_from_s3(bucket_name, image_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    return response["Body"].read()  # Return raw image bytes


def get_metadata_from_csv(bucket_name, csv_key, image_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
    metadata_df = pd.read_csv(io.BytesIO(response["Body"].read()))

    if "rsn" not in metadata_df.columns:
        raise KeyError("ERROR: Column 'rsn' not found in metadata file.")

    rsn = os.path.basename(image_key).split(".")[0]

    matching_row = metadata_df[metadata_df["rsn"] == rsn]
    return matching_row.iloc[0].to_dict() if not matching_row.empty else {}


def get_alt_text_from_s3(bucket_name, alt_text_folder, image_key):
    json_key = (
        f"{alt_text_folder}/{os.path.basename(image_key).split('.')[0]}_alt_text.json"
    )
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=json_key)
        alt_text_data = json.loads(response["Body"].read().decode("utf-8"))
        return alt_text_data.get("alt_text", "No alt-text available.")
    except s3_client.exceptions.NoSuchKey:
        return "No alt-text available."


def test_random_image():
    # Pick a random image from S3
    image_key = get_random_image(bucket_name, image_folder)
    if not image_key:
        print(" No images found in S3 bucket.")
        return

    metadata = get_metadata_from_csv(bucket_name, csv_key, image_key)
    title = metadata.get("title", "Unknown Title")
    caption = metadata.get("caption", "No caption available.")

    alt_text = get_alt_text_from_s3(bucket_name, alt_text_folder, image_key)

    image_data = download_image_from_s3(bucket_name, image_key)
    display(IPImage(data=image_data))

    print(f"\n **Image Key:** {image_key}")
    print(f"\n **Title:** {title}")
    print(f"\n **Caption:** {caption}")
    print(f"\n **Generated Alt-Text:**\n{alt_text}")


def test_random_image_return():
    # Pick a random image from S3
    image_key = get_random_image(bucket_name, image_folder)
    if not image_key:
        print(" No images found in S3 bucket.")
        return

    metadata = get_metadata_from_csv(bucket_name, csv_key, image_key)
    title = metadata.get("title", "Unknown Title")
    caption = metadata.get("caption", "No caption available.")

    alt_text = get_alt_text_from_s3(bucket_name, alt_text_folder, image_key)

    # image_data = download_image_from_s3(bucket_name, image_key)
    # display(IPImage(data=image_data))

    return {
        "image_key": image_key,
        "title": title,
        "caption": caption,
        "alt_text": alt_text,
    }


if __name__ == "__main__":
    test_random_image()
