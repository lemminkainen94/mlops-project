import json
import os
import pickle
import sys

import pandas as pd

from google.cloud import storage
from sklearn.linear_model import SGDClassifier


DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')


def download_files_gcs():
    storage_client = storage.Client()

    bucket = storage_client.bucket(DATA_BUCKET)

    preprocessors_blob = bucket.blob("model/preprocessors.pkl")
    model_blob = bucket.blob("model/model.pkl")

    preprocessors_blob.download_to_filename("/tmp/preprocessors.pkl")
    model_blob.download_to_filename("/tmp/model.pkl")


def get_model_artifacts():
    with open("/tmp/preprocessors.pkl", "rb") as file:
        (cvect, tfidf) = pickle.load(file)

    return cvect, tfidf


def data_prep(df, tfidf, cvect):
    x = tfidf.transform(cvect.transform(df.target_text))
    return x


def load_model():
    with open("/tmp/model.pkl", "rb") as file:
        return pickle.load(file)


def endpoint(event, context):
    download_files_gcs()
    df = pd.read_csv(f'gs://{DATA_BUCKET}/data/data_future.tsv', sep='\t')
    
    cvect, tfidf = get_model_artifacts()
    X = data_prep(df, tfidf, cvect)

    model = load_model()
    df['prediction'] = model.predict(X)
    df.to_csv(f'gs://{DATA_BUCKET}/data/prediction.tsv', sep='\t')
