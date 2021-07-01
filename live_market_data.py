import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

data = yf.download(tickers='UBER', period='1d', interval='1m')

fig = go.Figure()

fig.add_trace(go.Candlestick(x=data.index, 
                             open=data['Open'], 
                             high=data['High'], 
                             low=data['Low'], 
                             close=data['Close'], 
                             name='market data'))

fig.update_layout(title='UBER live share price evolution', 
                  yaxis_title='Stock Price (USD per Shares)')

fig.update_xaxes(rangeslider_visible=True, 
                 rangeselector=dict(
                     buttons=list([
                         dict(count=15, label='15m', step='minute', stepmode='backward'),
                         dict(count=45, label='45m', step='minute', stepmode='backward'),
                         dict(count=1, label='HTD', step='hour', stepmode='todate'),
                         dict(count=2, label='2h', step='hour', stepmode='backward'),
                         dict(step='all')
                                 ])
                                   )
                 )

fig.show()