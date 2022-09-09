import json
import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, request

from scipy.stats import beta


SQL = 'SELECT * FROM reviews WHERE review_text LIKE %(food)s'


with open('../db_con_creds.json') as f:
    creds = json.load(f)

connection = psycopg2.connect(
    user=creds['user'],
    password=creds['pwd'],
    host=creds['host'],
    database=creds['db']
)


def food_query_db(food):
    cur = connection.cursor()
    cur.execute(SQL, {'food': f'%{food}%'})
    
    cols = ['id', 'name', 'city', 'category', 'stars', 'published_at', 'review_text', 'review_id']
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    df.published_at = df.published_at.apply(pd.Timestamp)
    df.review_text = df.review_text.fillna('').apply(lambda x: x.lower())

    cur.close()

    return df


def get_sentiment(stars):
    if stars < 3:
        return -1
    if stars > 3:
        return 1
    return 0


def get_rank(name_df, prior=2, top_res=10):
    name_df['sentiment'] = name_df.stars.apply(get_sentiment)

    rests = []

    for res in name_df.name.unique():
        pos_count = len(name_df[(name_df.name == res) & (name_df.sentiment == 1)])
        neg_count = len(name_df[(name_df.name == res) & (name_df.sentiment == -1)])
        neutral_count = len(name_df[(name_df.name == res) & (name_df.sentiment == 0)])
        # sample from beta distr to get a score
        # for each restaurant with at least one review mentioning the food
        rests.append({
            'name': res,
            'total reviews': pos_count + neg_count + neutral_count,
            '% positive': f"{pos_count / (pos_count + neg_count + neutral_count):.0%}",
            'score': beta.rvs(prior + pos_count, prior + neg_count)
        })

    rests.sort(key=lambda x: x['score'], reverse=True)
    return rests[:top_res]


app = Flask('food-reviews')


@app.route('/foodReviews', methods=['POST'])
def get_top_places(top_res=10, prior=2):
    content = request.json

    if 'mode' in content and content['mode'] == 'conservative':
        prior = 5
    if 'top_res' in content:
        top_res = int(top_res)

    df = food_query_db(content['food'])

    return json.dumps(get_rank(df, top_res=top_res, prior=prior))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    connection.close()
