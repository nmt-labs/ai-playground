import pandas as pd
from chefboost import Chefboost as chef

df = pd.read_csv("../sample-data/risco.csv", sep=',')
config = {'algorithm': 'ID3'}
model = chef.fit(df, config = config, target_label = 'Decision')

config = {'algorithm': 'C4.5'}
model = chef.fit(df, config = config, target_label = 'Decision')

config = {'algorithm': 'CART'}
model = chef.fit(df, config = config, target_label = 'Decision')