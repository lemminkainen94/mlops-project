#!/usr/bin/env bash

export DATA_BUCKET='wojtek-ml-project'
gcloud functions deploy endpoint \
    --trigger-bucket $DATA_BUCKET \
    --runtime python39