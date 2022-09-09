import os
import requests
import pandas as pd

from sklearn.metrics import f1_score


url = 'http://localhost:9696/tbsa'
DATA_BUCKET = os.getenv("DATA_BUCKET")


def get_prediction(text, target):
    res = requests.post(url, json={"target": target, "text": text})
    if res.ok:
        print(res.json())
    return res.json()['sentiment']


if __name__ == "__main__":
    df = pd.read_csv(f'gs://{DATA_BUCKET}/data/data_future.tsv', sep='\t')
    df['prediction'] = 0
    for i, row in df.iterrows():
        df.at[i, 'prediction'] = get_prediction(row.target, row.text)
    print(df.prediction.value_counts())
    print(f1_score(df.sentiment, df.prediction, average='macro'))
    df.to_csv(f'gs://{DATA_BUCKET}/data/predictions.tsv', sep='\t')
