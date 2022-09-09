import requests
import pandas as pd
import argparse

parser = argparse.ArgumentParser("foodReviews")
parser.add_argument('food', type=str, help='food type or meal you wish to eat')
parser.add_argument('--mode', type=str, default='normal', help='Two possible modes:\n' +
                    'normal: will allow less popular places with fewer opinions score better\n' +
                    'conservative: less popular places unlikely to make it, e.g. a place with 2 positive\n' +
                    'and 0 negative is higly unlikely to win with a place with 95 positive and 5 negative reviews\n' +
                    'default is normal, but pick conservative if you wish to give less popular places a shot.')
parser.add_argument('--top', type=int, default=10, help='How many top results you wish to receive')

url = 'http://localhost:9696/foodReviews'

args = parser.parse_args()
print(args)
res = requests.post(url, json={"food": args.food, "mode": args.mode, "top_res": args.top})
if res.ok:
    print(pd.DataFrame(res.json()))
