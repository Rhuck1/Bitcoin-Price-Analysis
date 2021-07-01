import pandas as pd
from pandas_datareader import data, wb
import datetime
import plotly.graph_objs as go

start = pd.to_datetime('2018-02-04')
end = pd.to_datetime('2020-05-29')

df = data.DataReader('J', 'yahoo', start, end)

df['Middle Band'] = df['Close'].rolling(window=20).mean()

df['Upper Band'] = df['Middle Band'] + 1.96 * df['Close'].rolling(window=20).std()
df['Lower Band'] = df['Middle Band'] - 1.96 * df['Close'].rolling(window=20).std()

fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'], line=dict(color='blue', width=0.7), name='Middle Band'))
fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], line=dict(color='red', width=1.5), name='Upper Band (Sell)'))
fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], line=dict(color='green', width=1.5), name='Lower Band (Buy)'))

fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='market data'))

fig.update_layout(title='Bollinger Band Strategy', yaxis_title='Jacobs Engineering Stock Price (USD per Shares)')

fig.update_xaxes(rangeslider_visible=True, 
                 rangeselector=dict(
                        buttons=list([
                                dict(count=1, label='1m', step='month', stepmode='backward'),
                                dict(count=6, label='6m', step='month', stepmode='backward'),
                                dict(count=1, label='YTD', step='year', stepmode='todate'),
                                dict(count=1, label='1y', step='year', stepmode='backward'),
                                dict(step='all')
                                    ])
                                    )
                )

fig.show()