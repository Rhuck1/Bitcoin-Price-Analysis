import pandas as pd
from pandas_datareader import data, wb
import datetime

start = pd.to_datetime('2020-02-04')
end = pd.to_datetime('today')

tesla_df = data.DataReader('TSLA', 'yhoo', start, end)