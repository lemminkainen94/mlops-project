"""
In order to simulate a real-life setting the data is split into current and future sets
and then splits the current set into train and test. Train set can be used for cross-validation
or split further into train and dev sets.
Writes the results to the data file, as well as a GCS bucket.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


load_dotenv()
BUCKET = os.getenv("BUCKET")

df = pd.read_csv('data/data_no_tweet.tsv', sep='\t')
df['target_text'] = df.apply(lambda x: x.text + ' ' + x.target, axis=1)
df.to_csv(f'gs://{BUCKET}/data/data_no_tweet.tsv', sep='\t')

df_curr, df_future = train_test_split(df, test_size=0.2)
df_future.to_csv(f'gs://{BUCKET}/data/data_future.tsv', sep='\t')

df_train, df_test = train_test_split(
    df_curr, test_size=0.2, stratify=df_curr.sentiment
)

df_train.to_csv(f'gs://{BUCKET}/data/data_train.tsv', sep='\t')
df_test.to_csv(f'gs://{BUCKET}/data/data_test.tsv', sep='\t')
