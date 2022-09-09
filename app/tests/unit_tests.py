import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from main import get_model_artifacts, data_prep, load_model


DATA_BUCKET = os.getenv("DATA_BUCKET", 'wojtek-ml-project')


def download_files():
    os.system("cp ./model/* /tmp")
    os.system("cp ./data/data_future.tsv /tmp")


def remove_files():
    os.system("rm /tmp/*.pkl")
    os.system("rm /tmp/data_future.tsv")


@pytest.mark.offline
def test_get_model_arifacts():
    download_files()
    cvect, tfidf = get_model_artifacts()
    assert (type(cvect) == CountVectorizer) and (type(tfidf) == TfidfTransformer)
    remove_files()

@pytest.mark.offline
def test_load_model():
    download_files()
    model = load_model()
    assert type(model) == SGDClassifier
    remove_files()


@pytest.mark.offline
def test_data_prep():
    download_files()
    df = pd.read_csv('/tmp/data_future.tsv', sep='\t')
    cvect, tfidf = get_model_artifacts()
    X = data_prep(df, tfidf, cvect)
    assert type(X) == scipy.sparse.csr.csr_matrix
    remove_files()
