from typing import Dict
import csv
import pandas as pd
from io import BytesIO, StringIO
from google.cloud import storage


def parse_target_format(tf: Dict):
    target_values = []
    mappings = tf["mappings"]
    for mapping in mappings:
        if mapping["type"] == "ONE_TO_ONE":
            target_values.append(mapping["target"])
            mapping["selector"] = ""
    return target_values, tf


def parse_csv_headers_local(file_path):
    try:
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            headers = next(reader)
            return headers
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except StopIteration:
        print(f"Error: CSV file is empty: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def parse_excel_headers_local(file_path):
    df = pd.read_excel(file_path, nrows=1)
    return df.columns.tolist()


def get_headers_from_gcs_excel(bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        content = blob.download_as_bytes()
        excel_file = BytesIO(content)

        df = pd.read_excel(excel_file, nrows=1)

        return df.columns.tolist()

    except Exception as e:
        print(f"Error reading Excel file from GCS: {e}")
        return None


def get_headers_from_gcs_csv(bucket_name, blob_name):
    try:
        storage_client = storage.Client()

        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        content = blob.download_as_string().decode("utf-8")
        csv_file = StringIO(content)

        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)

        return headers

    except Exception as e:
        print(f"Error reading CSV file from GCS: {e}")
        return None
