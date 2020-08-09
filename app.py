"""
Process-sleep-data.py
Processes fitbit sleep data files, uploads data into fitbit dataframe, outputs graphs.
"""  	

import pandas as pd
import numpy as np
import datetime as dt  
import matplotlib.pyplot as plt  
from datetime import datetime                          
import streamlit as st
try: 
    json_normalize = pd.json_normalize
except:
    from pandas.io.json import json_normalize

try:
    import plotly.express as px
    import plotly.graph_objects as go # or plotly.express as px

    PLOTLY = True
except:
    PLOTLY = False
from tqdm.auto import tqdm
import seaborn as sns    
from statsmodels.tsa.stattools import grangercausalitytests

# Function: process_fitbit_sleep_data()
# fileList: A list of fitbit sleep data files eg ["sleep-2020-03-09.json","sleep-2020-04-08.json".....]
# Returns a dataframe with the following columns:
# ['duration', 'efficiency', 'endTime', 'mainSleep', 'minutesAfterWakeup', 'minutesAsleep', 'minutesAwake', 'minutesToFallAsleep', 'startTime', 'summary.asleep.count', 'summary.asleep.minutes', 'summary.awake.count', 'summary.awake.minutes', 'summary.deep.count', 'summary.deep.minutes', 'summary.deep.thirtyDayAvgMinutes', 'summary.light.count', 'summary.light.minutes', 'summary.light.thirtyDayAvgMinutes', 'summary.rem.count', 'summary.rem.minutes', 'summary.rem.thirtyDayAvgMinutes', 'summary.restless.count', 'summary.restless.minutes', 'summary.wake.count', 'summary.wake.minutes', 'summary.wake.thirtyDayAvgMinutes', 'timeInBed', 'type', 'dayOfWeek', 'rem.%', 'deep.%', 'wake.%', 'light.%', 'startMin', 'endMin']



#@st.cache            
def process_fitbit_sleep_data(fileList):
    full_sleep_df = None
    for input_file in fileList:#,title='Loading in fitbit data'):
        input_df = pd.read_json(input_file)
        detail_df = json_normalize(input_df['levels'])
        sleep_df = pd.concat([input_df, detail_df], axis =1)
        full_sleep_df = pd.concat([full_sleep_df, sleep_df], sort=True)

    full_sleep_df['dateOfSleep']= pd.to_datetime(full_sleep_df['dateOfSleep'])
    full_sleep_df['dayOfWeek'] = full_sleep_df['dateOfSleep'].dt.day_name()
    full_sleep_df = full_sleep_df.set_index('dateOfSleep')
    full_sleep_df.sort_index(inplace=True)

    full_sleep_df['duration'] = full_sleep_df['duration']/(1000*60) # convert duration to minutes

    for col in ['rem','deep','wake','light']:
        full_sleep_df[col + '.%'] = 100*full_sleep_df['summary.' + col + '.minutes']/full_sleep_df['duration']

    full_sleep_df['startMin'] = pd.to_datetime(full_sleep_df['startTime']).dt.minute + 60 * pd.to_datetime(full_sleep_df['startTime']).dt.hour

    full_sleep_df['startMin'] = np.where(full_sleep_df['startMin'] < 240, full_sleep_df['startMin'] + 1440, full_sleep_df['startMin']) # handle v late nights

    full_sleep_df['endMin'] = pd.to_datetime(full_sleep_df['endTime']).dt.minute + 60 * pd.to_datetime(full_sleep_df['endTime']).dt.hour

    #remove rows which are not mainSleep == True (these are naps not sleeps)
    full_sleep_df = full_sleep_df[full_sleep_df.mainSleep != False]

    #remove column which are not needed/useful
    full_sleep_df.drop(['logId', 'data', 'shortData', 'infoCode', 'levels'], axis=1, inplace=True)

    return full_sleep_df

# Function: plot_fitbit_sleep_data()
# sleep_df: a sleep dataframe
# cols: the columns to be displayed on the line graph
# output: saves a line graph to a png file named according to the columns graphed (with vertical lines to indicate weekends)
def plot_fitbit_sleep_data(sleep_df, cols):
    sleep_df[cols].plot(figsize=(20,5))
    plt.title('Sleep plot')

    for date in sleep_df.index[sleep_df['dayOfWeek'] == 'Monday'].tolist():
        plt.axvline(date)

    output_filename = "sleep"
    for col in cols:
        output_filename += "-" + col
    
    output_filename += ".png"
	
    st.pyplot()
def plot_fitbit_sleep_data_plotly(sleep_df, cols):

    X = [i for i in range(0,len(sleep_df.index.values))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=sleep_df['rem.%'],
                        mode='lines',
                        name='REM sleep'))
    fig.add_trace(go.Scatter(x=X, y=sleep_df['deep.%'],
                        mode='lines+markers',
                        name='deep sleep'))
    st.write(fig)

# Function: plot_sleep_data_scatter 
# sleep_df: a sleep dataframe
# cols: the columns to be displayed on the scatter plot
# output: saves a scatter plot to a png file named according to the columns graphed
def plot_sleep_data_scatter(full_sleep_df, col1, col2):
    full_sleep_df.plot.scatter(x=col1, y=col2)
    plt.title('Sleep plot')
    plt.ylabel(col2)
    plt.xlabel(col1)
    st.pyplot()

def plot_sleep_data_scatter_plotly(full_sleep_df, col1, col2):
    fig = px.scatter(x=full_sleep_df[col1], y=full_sleep_df[col2],labels={
                     "x": col1,
                     "y": col2                 
                },
                title="Sleep plot")

    st.write(fig)

# Function: plot_corr(sleep_df)
# Function plots a graphical correlation matrix for each pair of columns in the dataframe.
# sleep_df: pandas DataFrame

def plot_corr(sleep_df):

    f = plt.figure(figsize=(19, 15))
    fig = sns.heatmap(sleep_df.corr())
    
    cols = ['duration', 'efficiency', 'summary.deep.minutes', 'summary.deep.minutes.%', 'summary.light.minutes', 'summary.light.minutes.%', 'summary.rem.minutes', 'summary.rem.minutes.%', 'summary.wake.minutes', 'summary.wake.minutes.%', 'startMin', 'avg4_startMin', 'startTimeDeviation1.%', 'startTimeDeviation4.%']
    plt.xticks(range(sleep_df.shape[1]), cols, fontsize=12, rotation="vertical")
    plt.yticks(range(sleep_df.shape[1]), cols, fontsize=12)

    plt.title('Correlation Matrix', fontsize=14)
    st.pyplot()
from sklearn.impute import SimpleImputer
import copy
def df_to_plotly(df,log=False):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def plot_corr_plotly(sleep_df):
    fig = go.Figure(data=go.Heatmap(df_to_plotly(sleep_df.corr())))
    st.write(fig)
def plot_df_plotly(sleep_df):
    fig = go.Figure(data=go.Heatmap(df_to_plotly(sleep_df,log=True)))
    st.write(fig)


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    Next :
    https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
def check_time_lags(df,col0,col1):
    d1 = df[col0]
    d2 = df[col1]
    hours = 5
    days = 30
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(hours*days),int(hours*days+1))]
    offset = np.ceil(len(rs)/2)-np.argmax(rs)
    f,ax=plt.subplots(figsize=(20,13))
    re_title = 'look for time lags when correlation is maximised (peak synchrony) betwen {0} and {1}'.format(col0,col1)
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads'+re_title,ylim=[.1,.31],xlim=[0,301], xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
    plt.legend()  
    st.pyplot()

#import statsmodels.api as sm
#from statsmodels.tsa.api import VAR
#def analyze_coherance(df,col0,col1):
#    d1 = df[col0]
#    d2 = df[col1]
    #plt.specgram(signalData,Fs=1)


    #plt.cohere(d1,d2,NFFT=len(d1)/2)  
    # import for Granger's Causality Test
#    granger_test = sm.tsa.stattools.grangercausalitytests(df, maxlag=2, verbose=True)
#    st.text(granger_test)
#    df_differenced = df.diff().dropna()

##    model = VAR(df_differenced)
#    results = model.fit(maxlags=15, ic='aic')
#    st.text(results.summary())
    #st.pyplot()



if __name__ == "__main__":  
    st.title('Analysis for sleep quality')
    fileList = ["sleep-2020-03-09.json","sleep-2020-04-08.json","sleep-2020-05-08.json","sleep-2020-06-07.json","sleep-2020-07-07.json"]

    st.markdown('''
    This is a markdown string that explains sleep data from date {0}
    '''.format(str('2020-03-09')))
    sleep_df = process_fitbit_sleep_data(fileList)
        #st.write(df1)
    df = copy.copy(sleep_df)
 
    del df['endTime']
    del df['dayOfWeek']
    del df['startTime']
    del df['type']
    #imp = SimpleImputer(strategy="most_frequent")
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df.values[:] = imp.fit_transform(df.values[:])

    #st.write(sleep_df, unsafe_allow_html=True)
    if PLOTLY:
        plot_fitbit_sleep_data_plotly(sleep_df, ['rem.%', 'deep.%'])
        plot_sleep_data_scatter_plotly(sleep_df, 'startMin', 'deep.%')
        plot_corr_plotly(sleep_df)
        check_time_lags(sleep_df,'rem.%','deep.%')
        plot_df_plotly(sleep_df)#,'rem.%','deep.%')
     

    else:
        plot_corr(sleep_df)
        plot_fitbit_sleep_data(sleep_df, ['rem.%', 'deep.%'])
        plot_sleep_data_scatter(sleep_df, 'startMin', 'deep.%')



#NFFT=256, Fs=2, Fc=0, detrend=, window=, noverlap=0, pad_to=None, sides=’default’, scale_by_freq=None, *, data=None, **kwargs)