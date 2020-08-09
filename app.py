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
# Function: process_fitbit_sleep_data()
# fileList: A list of fitbit sleep data files eg ["sleep-2020-03-09.json","sleep-2020-04-08.json".....]
# Returns a dataframe with the following columns:
# ['duration', 'efficiency', 'endTime', 'mainSleep', 'minutesAfterWakeup', 'minutesAsleep', 'minutesAwake', 'minutesToFallAsleep', 'startTime', 'summary.asleep.count', 'summary.asleep.minutes', 'summary.awake.count', 'summary.awake.minutes', 'summary.deep.count', 'summary.deep.minutes', 'summary.deep.thirtyDayAvgMinutes', 'summary.light.count', 'summary.light.minutes', 'summary.light.thirtyDayAvgMinutes', 'summary.rem.count', 'summary.rem.minutes', 'summary.rem.thirtyDayAvgMinutes', 'summary.restless.count', 'summary.restless.minutes', 'summary.wake.count', 'summary.wake.minutes', 'summary.wake.thirtyDayAvgMinutes', 'timeInBed', 'type', 'dayOfWeek', 'rem.%', 'deep.%', 'wake.%', 'light.%', 'startMin', 'endMin']


from tqdm.auto import tqdm
import streamlit as st


class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)
def process_fitbit_sleep_data(fileList):
    full_sleep_df = None
    for input_file in tqdm(fileList,title='Loading in fitbit data'):
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
    #plt.savefig(output_filename, dpi=100)
    #plt.close()	
    st.pyplot()


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
    #output_filename = "sleep-scatter-" + col1 + "-" + col2 + ".png"
    #plt.savefig(output_filename, dpi=100)
    #plt.close()	

# Function: plot_corr(sleep_df)
# Function plots a graphical correlation matrix for each pair of columns in the dataframe.
# sleep_df: pandas DataFrame

def plot_corr(sleep_df):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(sleep_df.corr(), fignum=f.number)
    cols = ['duration', 'efficiency', 'summary.deep.minutes', 'summary.deep.minutes.%', 'summary.light.minutes', 'summary.light.minutes.%', 'summary.rem.minutes', 'summary.rem.minutes.%', 'summary.wake.minutes', 'summary.wake.minutes.%', 'startMin', 'avg4_startMin', 'startTimeDeviation1.%', 'startTimeDeviation4.%']
    plt.xticks(range(sleep_df.shape[1]), cols, fontsize=12, rotation="vertical")
    plt.yticks(range(sleep_df.shape[1]), cols, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.title('Correlation Matrix', fontsize=14)
    st.pyplot()

    # corr = sleep_df.corr()
    # size = 10
    # fig, ax = plt.subplots(figsize=(size, size))
    # ax.matshow(corr)
    # plt.yticks(range(len(corr.columns)), corr.columns);
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    #output_filename = "full-sleep-correlation-matrix.png"
    #plt.savefig(output_filename, dpi=100)
    #plt.close()	

#if __name__ == "__main__":  
st.title('fitbit analysis for sleep')

fileList = ["sleep-2020-03-09.json","sleep-2020-04-08.json","sleep-2020-05-08.json","sleep-2020-06-07.json","sleep-2020-07-07.json"]
st.markdown('''
This is a markdown string that explains sleep data from date {0}
'''.format(str('2020-03-09')))
sleep_df = process_fitbit_sleep_data(fileList)


plot_fitbit_sleep_data(sleep_df, ['rem.%', 'deep.%'])

plot_sleep_data_scatter(sleep_df, 'startMin', 'deep.%')
plot_corr(sleep_df)
st.write(sleep_df, unsafe_allow_html=True)


