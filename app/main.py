"""
Main application.
Downloads the deployed model to run predictions against a new dataset.
Writes the predictions to cloud storage.
"""
import os
import pickle

import pandas as pd

from dotenv import load_dotenv
from google.cloud import storage


load_dotenv()

BUCKET = os.getenv("BUCKET")


def download_files_gcs():
    """download preprocessors and the deployed model from cloud to tmp folder"""
    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET)

    preprocessors_blob = bucket.blob("model/preprocessors.pkl")
    model_blob = bucket.blob("model/model.pkl")

    preprocessors_blob.download_to_filename("/tmp/preprocessors.pkl")
    model_blob.download_to_filename("/tmp/model.pkl")


def get_model_artifacts():
    """gets data preprocessors. fit to the training data"""
    with open("/tmp/preprocessors.pkl", "rb") as file:
        (cvect, tfidf) = pickle.load(file)

    return cvect, tfidf


def data_prep(data, tfidf, cvect):
    """prepares the data for prediction using tfidf transform"""
    tfidf_data = tfidf.transform(cvect.transform(data.target_text))
    return tfidf_data


def load_model():
    """loads the deployed model"""
    with open("/tmp/model.pkl", "rb") as file:
        return pickle.load(file)


def endpoint(event, context):
    """google cloud funciton; triggered by uploading new data to the cloud storage"""
    print(event, context)
    download_files_gcs()
    df_future = pd.read_csv(f'gs://{BUCKET}/data/data_future.tsv', sep='\t')

    cvect, tfidf = get_model_artifacts()
    data = data_prep(df_future, tfidf, cvect)

    model = load_model()
    df_future['prediction'] = model.predict(data)
    df_future.to_csv(f'gs://{BUCKET}/data/prediction.tsv', sep='\t')
