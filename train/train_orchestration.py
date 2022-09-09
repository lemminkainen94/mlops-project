"""
Uses hyperopt to search for the best model, then logs the best model. Builds a Prefect Deployment
"""
import os
import time
import warnings

import pickle

import mlflow
import numpy as np
import pandas as pd

from datetime import timedelta
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect.blocks.system import String
from prefect import flow, task
from prefect.deployments import Deployment

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

TBSA_EXPERIMENT_NAME = "tbsa-train"
EXPERIMENT_NAME = "chosen-models-tbsa"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(TBSA_EXPERIMENT_NAME)


@task
def save_preprocessors(count_vect, tfidf):
    with open("preprocessors.pkl", "wb") as file:
        pickle.dump((count_vect, tfidf), file)


@task
def prepare_data(df_train, df_test, vectorizer_params):
    count_vect = CountVectorizer(**vectorizer_params)
    word_idx = count_vect.fit_transform(df_train.target_text.values)

    tfidf_transformer = TfidfTransformer().fit(word_idx)

    x_train = tfidf_transformer.transform(word_idx)
    x_test = tfidf_transformer.transform(count_vect.transform(df_test.target_text.values))
    
    return x_train, x_test, df_train.sentiment, df_test.sentiment, count_vect, tfidf_transformer


@task
def hyperopt_search(x_train, y_train):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag('model', 'sgd')
            mlflow.log_params(params)
            sgd = SGDClassifier(max_iter=1000, **params)
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            f1_mac = cross_val_score(sgd, x_train, y_train, cv=kfold, scoring='f1_macro', verbose=False).mean()
            mlflow.log_metric('f1_macro', f1_mac)
        
        return {'loss': -f1_mac, 'status': STATUS_OK}
    
    loss = ['hinge', 'log', 'perceptron', 'modified_huber']
    penalty = ['l1', 'l2']

    search_space = {
        'loss': hp.choice('loss', loss),
        'alpha': hp.loguniform('alpha', -8, -1),
        'penalty': hp.choice('penalty', penalty),
        'tol': hp.loguniform('tol', -4, 0)
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )

    best_result['loss'] = loss[best_result['loss']]
    best_result['penalty'] = penalty[best_result['penalty']]
    print(best_result)


@task
def train_best_model(x_train, y_train, x_test, y_test, tag, params):
    with mlflow.start_run():
        mlflow.set_tag("version_tag", tag)
        sgd = SGDClassifier(max_iter=1000, **params)
        sgd.fit(x_train, y_train)

        predicted = sgd.predict(x_test)
        f1_mac = f1_score(y_test, predicted, average='macro') 
        acc = accuracy_score(y_test, predicted)
        mlflow.log_metric("f1_macro", f1_mac)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact("preprocessors.pkl", artifact_path="model")


@flow
def train_and_log_best(x_train, y_train, x_test, y_test):
    try:
        current_tag_block = String.load("version-counter")
    except ValueError:
        current_tag_block = String(value='1')
        current_tag_block.save(name="version-counter")
    print(current_tag_block)
    current_tag = int(current_tag_block.value)

    # retrieve the best model and log it to MLflow
    experiment = client.get_experiment_by_name(TBSA_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_macro DESC"],
    )
    params = best_run[0].data.params
    params['alpha'] = float(params['alpha'])
    params['tol'] = float(params['tol'])
    train_best_model(x_train, y_train, x_test, y_test, current_tag, params)
    os.system("rm preprocessors.pkl")
    
    experiment_id = client.get_experiment_by_name(TBSA_EXPERIMENT_NAME).experiment_id
    all_runs = client.search_runs(experiment_ids=experiment_id)
    for mlflow_run in all_runs:
        client.delete_run(mlflow_run.info.run_id)
    
    new_tag = String(value=f"{current_tag + 1}")
    new_tag.save(name="version-counter", overwrite=True)


@flow
def main():
    df_train = pd.read_csv(f'gs://{DATA_BUCKET}/data/data_train.tsv', sep='\t')
    df_test = pd.read_csv(f'gs://{DATA_BUCKET}/data/data_test.tsv', sep='\t')
    vectorizer_params = dict(ngram_range=(1, 2), max_df=0.8)
    x_train, x_test, y_train, y_test, cvect, tfidf_trans = prepare_data(df_train, df_test, vectorizer_params)
    save_preprocessors(cvect, tfidf_trans)
    hyperopt_search(x_train, y_train)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    train_and_log_best(x_train, y_train, x_test, y_test)


deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    work_queue_name="ml",
)

deployment.apply()