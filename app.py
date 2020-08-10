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
#from statsmodels.tsa.stattools import grangercausalitytests
import plotting_functions as pf
# Function: process_fitbit_sleep_data()
# fileList: A list of fitbit sleep data files eg ["sleep-2020-03-09.json","sleep-2020-04-08.json".....]
# Returns a dataframe with the following columns:
# ['duration', 'efficiency', 'endTime', 'mainSleep', 'minutesAfterWakeup', 'minutesAsleep', 'minutesAwake', 'minutesToFallAsleep', 'startTime', 'summary.asleep.count', 'summary.asleep.minutes', 'summary.awake.count', 'summary.awake.minutes', 'summary.deep.count', 'summary.deep.minutes', 'summary.deep.thirtyDayAvgMinutes', 'summary.light.count', 'summary.light.minutes', 'summary.light.thirtyDayAvgMinutes', 'summary.rem.count', 'summary.rem.minutes', 'summary.rem.thirtyDayAvgMinutes', 'summary.restless.count', 'summary.restless.minutes', 'summary.wake.count', 'summary.wake.minutes', 'summary.wake.thirtyDayAvgMinutes', 'timeInBed', 'type', 'dayOfWeek', 'rem.%', 'deep.%', 'wake.%', 'light.%', 'startMin', 'endMin']


if __name__ == "__main__":  
    st.title('Analysis for sleep quality')
    fileList = ["sleep-2020-03-09.json","sleep-2020-04-08.json","sleep-2020-05-08.json","sleep-2020-06-07.json","sleep-2020-07-07.json"]

    st.markdown('''
    This is a markdown string that explains sleep data from date {0}
    '''.format(str('2020-03-09')))
    sleep_df = pf.process_fitbit_sleep_data(fileList)


    if PLOTLY:

        pf.plot_fitbit_sleep_data_plotly(sleep_df, ['rem.%', 'deep.%'])
        pf.plot_sleep_data_scatter_plotly(sleep_df, 'startMin', 'deep.%')

        '''
        If imputation worked there would be no nans visible below:
        need to drop nans
        fill with zero,
        impute zeros with mean
        '''
        #pf.plot_df_plotly(sleep_df)#,'rem.%','deep.%')
        pf.plot_corr(sleep_df)

        pf.plot_corr_plotly(sleep_df)

        #NON_DER = ['startMin',]
        #ts_sleep_df = pf.df_derived_by_shift(sleep_df,1,NON_DER)
        #pf.plot_corr(ts_sleep_df)
        #pf.covariance_matrix(sleep_df)

        pf.check_time_lags(sleep_df,'rem.%','deep.%')
        



    else:
        pf.plot_corr(sleep_df)
        pf.plot_fitbit_sleep_data(sleep_df, ['rem.%', 'deep.%'])
        pf.plot_sleep_data_scatter(sleep_df, 'startMin', 'deep.%')



#NFFT=256, Fs=2, Fc=0, detrend=, window=, noverlap=0, pad_to=None, sides=’default’, scale_by_freq=None, *, data=None, **kwargs)