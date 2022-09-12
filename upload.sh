export $(cat .env)

gsutil -m cp model/* gs://$BUCKET/model/
gsutil -m cp data/*.tsv gs://$BUCKET/data/
