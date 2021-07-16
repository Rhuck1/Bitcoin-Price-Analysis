import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# statsmodels.tsa.filters.hp_filter.hpfilter
# statsmodels.tsa.seasonal.STL

def import_test():
    print('import test successful')

def test_stationarity(dataframe, model=adfuller):
    '''Input: timeseries dataframe of size (n, )
              desired statsmodel model
                            
       Output: Model results and graph of original timeseries dataframe, moving AVG, moving STD
    '''
    
    # Determining rolling statistics
    movingAverage = dataframe.rolling(window=12).mean()
    movingSTD = dataframe.rolling(window=12).std()

    # Plot rolling statistics
    fig = go.Figure()
    
    fig.add_trace(go.Line(x=dataframe.index, y=dataframe, line=dict(color='blue'), name='Original'))
    fig.add_trace(go.Line(x=dataframe.index, y=movingAverage, line=dict(color='orange'), name='Rolling Mean'))
    fig.add_trace(go.Line(x=dataframe.index, y=movingSTD, line=dict(color='aqua'), name='Rolling STD'))
    
    fig.update_layout(title='Rolling Mean and Standard Deviation', template='none')
    fig.show()
    
    models = {
        adfuller: 'Dickey-Fuller'
    }
    
#     # Perform Dickey-Fuller Test:
    print(f'Results of {models[model]} Test:')
    dftest = model(dataframe, autolag='AIC')
    dfoutput = pd.Series(dftest[0: 4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    print(dfoutput)


    
def decomposition_components(dataframe):
    
    decomposition = seasonal_decompose(dataframe)
    
    original = dataframe
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Plot decomposition components
    for plot in ['Original', 'Trend', 'Seasonal', 'Residual']:

        plots = {
            'Original': original,
            'Trend': trend,
            'Seasonal': seasonal,
            'Residual': residual
        }

        titles = {
            'Original': 'Price',
            'Trend': 'Trend',
            'Seasonal': 'Seasonal',
            'Residual': 'Residual'
        }

        fig = px.line(x=dataframe.index, y=plots[plot], template='none')
        fig.update_layout(title=titles[plot])

        fig.show()


        
def plot_acf_pacf(dataframe, n_lags):
    
    lag_acf = acf(dataframe, nlags=n_lags, fft=False)
    lag_pacf = pacf(dataframe, nlags=n_lags, method='ols')
    
    plots = {
        'Autocorrelation Function': lag_acf,
        'Partial Autocorrelation Function': lag_pacf
    }
    
    # Plot ACF and PACF
    for title in plots.keys():
        
        fig = px.line(y=plots[title], template='none')
        fig.add_hline(y=0, line_dash='dash', line_color='grey')
        
        #  Are hese are supposed to be confidence intervals, sigma (STD of data) * 1.96???
        fig.add_hline(y=-1.96 / np.sqrt(len(dataframe)), line_dash='dash', line_color='grey', annotation_text="Confidence Interval", 
              annotation_position="bottom right")
        fig.add_hline(y=1.96 / np.sqrt(len(dataframe)), line_dash='dash', line_color='grey', annotation_text="Confidence Interval", 
              annotation_position="upper right")
        
        fig.update_xaxis(title='Number of Lags')

        fig.update_layout(title=title)
        
        fig.show()        


        
def get_hurst_exponent(time_series, max_lag=20):
    '''Returns the Hurst Exponent of the time series'''
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]        
        
        
                       
def timestep_creator(dataframe, timesteps=60, reshape=False):
    '''Input: dataframe or nd.array of size (n, 1)
              number of timesteps (aka len(array)) samples from the dataframe
              reshape arrays to size (n, 1, 1) for purposes of LSTM ML 3D input requirements
              
       Output: X_train, y_train timestep sample arrays each of len = n-timesteps, array size = (n, 1) or (n, 1, 1)    
    '''
      
    X_train = []
    y_train = []
    
    if type(dataframe) == np.ndarray:
        
        for i in range(timesteps, dataframe.shape[0]):
            
            X_train.append(dataframe[i - timesteps: i, 0])
            y_train.append(dataframe[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
    
    else:
        
        for i in range(timesteps, dataframe.shape[0]):

            X_train.append(dataframe.iloc[i - timesteps: i, 0])
            y_train.append(dataframe.iloc[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
    
    if reshape:

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))

    return X_train, y_train



def rolling_forecast_origin(train, min_train_size, horizon):
    '''Rolling Forecast Origin Generator
       Stride is inherently 1    
    '''
    
    for i in range(len(train) - min_train_size - horizon + 1):
        
        split_train = train[: min_train_size + i]
        split_val = train[min_train_size + i: min_train_size + i + horizon]
        
        yield split_train, split_val
        

        
def sliding_window(train, window_size, horizon):
    '''Sliding window generator
       Stride is inherently 1
    '''
    
    for i in range(len(train) - window_size - horizon + 1):
        
        split_train = train[i: window_size + i]
        split_val = train[i + window_size: window_size + i + horizon]
        
        yield split_train, split_val