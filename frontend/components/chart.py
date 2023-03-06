from dash import Dash, Input, Output, dcc, html

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dcc, html


API_URL = 'https://min-api.cryptocompare.com/data/histoday'

symbol = 'BTC'
limit = 365

params = {
    'fsym': symbol,
    'tsym': 'USD',
    'limit': limit,
}
response = requests.get(API_URL, params=params)
data = response.json()['Data']
df = pd.DataFrame(data, columns=[
    'time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto'])
df['time'] = pd.to_datetime(df['time'], unit='s')

# replace with your own data source
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
))


def chart():
    return dcc.Graph(id="graph", figure=fig, style={"height": "90vh"}),
