export DATA_BUCKET='wojtek-ml-project'
gsutil -m cp data/*.tsv gs://$DATA_BUCKET/data/
