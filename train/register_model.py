import os
import pickle
import sys

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import mlflow

DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = 'chosen-models-tbsa'
MODEL_NAME = 'tbsa-model'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)



def prepare_data(df, cvect, tfidf):
    word_idx = cvect.transform(df.target_text.values)
    x_test = tfidf.transform(word_idx)    
    return x_test, df_test.sentiment


run_id = sys.argv[1]
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/preprocessors.pkl", dst_path="./"
)

with open("preprocessors.pkl", "rb") as file:
    (cvect, tfdif) = pickle.load(file)

df_test = pd.read_csv(f'gs://{DATA_BUCKET}/data/data_test.tsv', sep='\t')
X_test, y_test = prepare_data(df_test, cvect, tfdif)
print(X_test.shape)

logged_model = f"runs:/{run_id}/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

y_pred = loaded_model.predict(X_test)
test_score = f1_score(y_test, y_pred, average='macro')

try:
    client.create_registered_model(name=MODEL_NAME)
except:
    pass

description = f"test score: {test_score}"
mv = client.create_model_version(
    name=MODEL_NAME, source=logged_model, run_id=run_id, description=description
)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="production",
    archive_existing_versions=True,
)

# download the model
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/preprocessors.pkl", dst_path="./model"
)
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/model.pkl", dst_path="./model"
)

# save the best model in a convenient place in your bucket for the google cloud function
os.system(f'gsutil -m cp model/* gs://{DATA_BUCKET}/model/')