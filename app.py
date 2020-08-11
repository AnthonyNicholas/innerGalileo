"""
Process-sleep-data.py
Processes fitbit sleep data files, uploads data into fitbit dataframe, outputs graphs.
"""  	

import pandas as pd
import numpy as np                       
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go # or plotly.express as px

    PLOTLY = True
except:
    PLOTLY = False

import sklearn   
import plotting_functions as pf
CACHED = True
if __name__ == "__main__":  
    if CACHED:
        st.title('The Quantified Sleep')

        fileList = ["sleep-2020-03-09.json","sleep-2020-04-08.json","sleep-2020-05-08.json","sleep-2020-06-07.json","sleep-2020-07-07.json"]
        st.markdown('Analysis for sleep quality')

        st.markdown('''
        This is a markdown string that explains sleep data fro
        m date {0}
        '''.format(str('2020-03-09')))
    else:
        print('add methods for user upload data')
    sleep_df = pf.process_fitbit_sleep_data(fileList)


    if PLOTLY:
        #st.markdown('''
        ## Using cached data for splash screen
        #Data cleaning follows
        #'''.format(np.sum(sleep_df.isnull().sum().values)))
        #st.markdown('''---''')

        #pf.plot_df_plotly(sleep_df)#,'rem.%','deep.%')
        sleep_df.dropna(axis=1, how='any',inplace=True, thresh=4)
        sleep_df = pf.try_to_impute(sleep_df)       
        sleep_df.dropna(axis=1, how='any',inplace=True, thresh=0)
        sleep_df.dropna(axis=0, how='any',inplace=True, thresh=10)
        sleep_df = sleep_df.apply(lambda x: x.fillna(sleep_df.mean()),axis=1)
        #sleep_df.dropna(axis=0,inplace=True)


        st.markdown('''---''')
        st.markdown('''\n\n''')
        st.markdown('''The slow snooze movement''')

        '''
        clustergram useful for exploration
        '''
        sleep_df.dropna(axis=0, how='any',inplace=True, thresh=4)
        reduced_df = pf.cluster_map_corr(sleep_df)
        reduced_df.values[:] = sklearn.preprocessing.normalize(reduced_df.values[:])
        '''
        Fully imputed data frame:
        '''
        st.markdown('''---''')
        st.markdown('''\n\n''')

        pf.plot_df_plotly(reduced_df)#,'rem.%','deep.%')
        sleep_df = reduced_df 
        st.markdown('''---''')
        st.markdown('''\n\n''')
        st.write(sleep_df.describe())

        st.markdown('''Total Nans: {}'''.format(np.sum(sleep_df.isnull().sum().values)))

        st.markdown('''\n\n''')

        pf.plot_fitbit_sleep_data_plotly(sleep_df, ['rem.%', 'deep.%'])
        st.markdown('''---''')
        st.markdown('''\n\n''')

        pf.plot_sleep_data_scatter_plotly(sleep_df, 'startMin', 'deep.%')
        st.markdown('''---''')

        '''
        # hexbins, 
        joint plot show density of data better:
        '''
        pf.plot_sleep_data_joint(sleep_df, 'startMin', 'deep.%')
    
        '''
        If imputation worked there would be no nans visible below:
        need to drop nans
        fill with zero,
        impute zeros with mean
        '''
        st.markdown('''---''')


        pf.plot_corr(sleep_df)
        ''' 
        This is plotly correlation matrix this can be combined with
        seaborn approach and masking
        '''
        st.markdown('''---''')
        reduced_df = pf.cluster_map_cov(sleep_df)

        pf.plot_imshow_plotly(sleep_df.corr())
        '''
        Covariance matrix backbone of PCA, more useful when we have more features
        variables that vary together includes negatively correlations, or non-proporttional changes
        big changes that correlate with small changes (sometimes negatively correlations)

        '''
        st.markdown('''---''')

        pf.covariance_matrix(sleep_df)
        st.markdown('''---''')
        pf.plot_imshow_plotly(sleep_df.cov())
        st.markdown('''---''')
        pf.check_time_lags(sleep_df, 'rem.%', 'deep.%')
        st.markdown('''---''')
        pf.check_time_lags(sleep_df, 'startMin', 'deep.%')




    else:
        pf.plot_corr(sleep_df)
        pf.plot_fitbit_sleep_data(sleep_df, ['rem.%', 'deep.%'])
        pf.plot_sleep_data_scatter(sleep_df, 'startMin', 'deep.%')



#NFFT=256, Fs=2, Fc=0, detrend=, window=, noverlap=0, pad_to=None, sides=’default’, scale_by_freq=None, *, data=None, **kwargs)