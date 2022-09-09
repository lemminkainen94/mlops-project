"""
Online Integration test
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from google.cloud import storage
from requests.packages.urllib3.util.retry import Retry

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from main import download_files_gcs, data_prep, load_model


DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')


@pytest.mark.online
def test_download_files_gcs():
    download_files_gcs()

    assert ("preprocessors.pkl" in os.listdir("/tmp") and "model.pkl" in os.listdir("/tmp"))

    os.system("rm -rf /tmp/*.pkl")


@pytest.mark.online
def test_function_online():
    os.system("mkdir tmp")
    storage_client_data = storage.Client()

    data_bucket = storage_client_data.bucket(DATA_BUCKET)

    data_blob = data_bucket.blob("data/prediction.csv")
    first_date = data_blob

    os.system("cd ../../ && ./upload.sh")

    data_blob_new = data_bucket.blob("data/prediction.csv")

    second_date = data_blob.updated
    os.system("rm -rf ./tmp/*.pkl")

    assert second_date != first_date
