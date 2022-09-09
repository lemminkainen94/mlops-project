"""
Prefect Flow for batch monitoring
"""
import json
import os
import pickle
import sys

import pandas as pd
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ClassificationPerformanceTab, DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    ClassificationPerformanceProfileSection,
    DataDriftProfileSection,
)
from pymongo import MongoClient
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from prefect import flow, task
from prefect.deployments import Deployment

DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')


def preprocess_data(df):
    with open("./model/preprocessors.pkl", "rb") as file:
        (cvect, tfidf) = pickle.load(file)

    x = tfidf.transform(cvect.transform(df.target_text.values))
    return x


@task
def load_reference_data(filename):
    with open('./model/model.pkl', 'rb') as f_in:
        sgd = pickle.load(f_in)

    reference_data = pd.read_csv(filename, sep='\t')
    x = preprocess_data(reference_data)
    reference_data['prediction'] = sgd.predict(x)
    return reference_data


@task
def fetch_data():
    df = pd.read_csv(f"gs://{DATA_BUCKET}/data/predictions.tsv", sep='\t')
    return df


@task
def run_evidently(ref_data, data):
    profile = Profile(
        sections=[DataDriftProfileSection(), ClassificationPerformanceProfileSection()]
    )
    mapping = ColumnMapping(
        prediction="prediction",
        target='sentiment'
    )
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(
        tabs=[ClassificationPerformanceTab(verbose_level=0)]
    )
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_html_report(result):
    result[1].save('./evidently_report.html')


@flow
def batch_analyze():
    ref_data = load_reference_data(f'gs://{DATA_BUCKET}/data/data_train.tsv')
    data = fetch_data()
    result = run_evidently(ref_data, data)
    save_html_report(result)


deployment = Deployment.build_from_flow(
    flow=batch_analyze,
    name="evidently-report",
    work_queue_name="ml",
)

deployment.apply()
