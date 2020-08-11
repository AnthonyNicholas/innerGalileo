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
    This is a markdown string that explains sleep data fro
    m date {0}
    '''.format(str('2020-03-09')))
    sleep_df = pf.process_fitbit_sleep_data(fileList)

    if PLOTLY:
        '''
        # To see nan's
        Mouse over 
        It looks like some whole columns are NaN and there are a few with just two or so samples.
        We should drop nan columns.
        '''

        pf.plot_df_plotly(sleep_df)#,'rem.%','deep.%')
        sleep_df.dropna(axis=1, how='any',inplace=True, thresh=4)
        sleep_df = pf.try_to_impute(sleep_df)
        #sleep_df.dropna(axis=0,inplace=True)

        '''
        # After drop nan
        '''
        pf.plot_df_plotly(sleep_df)#,'rem.%','deep.%')
        st.write(sleep_df.describe())
        '''
        clustergram useful for exploration
        '''
        #pf.cluster_map(sleep_df)
        pf.plot_fitbit_sleep_data_plotly(sleep_df, ['rem.%', 'deep.%'])
        pf.plot_sleep_data_scatter_plotly(sleep_df, 'startMin', 'deep.%')
        '''
        hexbins show density of data better:
        '''
        pf.plot_sleep_data_joint(sleep_df, 'startMin', 'deep.%')
    
        '''
        If imputation worked there would be no nans visible below:
        need to drop nans
        fill with zero,
        impute zeros with mean
        '''

        pf.plot_corr(sleep_df)
        ''' 
        This is plotly correlation matrix this can be combined with
        seaborn approach and masking
        '''
        pf.plot_imshow_plotly(sleep_df.corr())
        '''
        Covariance matrix backbone of PCA, more useful when we have more features
        variables that vary together includes negatively correlations, or non-proporttional changes
        big changes that correlate with small changes (sometimes negatively correlations)

        '''
        pf.covariance_matrix(sleep_df)
        pf.plot_imshow_plotly(sleep_df.cov())
        pf.check_time_lags(sleep_df, 'rem.%', 'deep.%')
        pf.check_time_lags(sleep_df, 'startMin', 'deep.%')




    else:
        pf.plot_corr(sleep_df)
        pf.plot_fitbit_sleep_data(sleep_df, ['rem.%', 'deep.%'])
        pf.plot_sleep_data_scatter(sleep_df, 'startMin', 'deep.%')



#NFFT=256, Fs=2, Fc=0, detrend=, window=, noverlap=0, pad_to=None, sides=’default’, scale_by_freq=None, *, data=None, **kwargs)