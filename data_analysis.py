#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler


import pickle # To save the model
from functools import partial
import random


from typing import List


# The following library is to plot the loss during training
# https://github.com/stared/livelossplot
get_ipython().system(' pip install livelossplot')
from livelossplot import PlotLossesKerasTF

from pandas.plotting import scatter_matrix # For plots
import matplotlib.pyplot as plt # For plots
get_ipython().run_line_magic('matplotlib', 'notebook')
# importing statistics module
import statistics
from scipy.stats import norm,pearsonr,spearmanr
import scipy.stats as st
from scipy.interpolate import interp1d, make_interp_spline


import os
from os.path import isfile


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


# In[2]:


# Read the full dataset
full_df = pd.read_csv('fastclick.csv')

# Remove the timestamp, which does not have predictive importance
full_df = full_df.drop(columns=["Unnamed: 621"])
full_df = full_df.drop(columns=["Time"])

data = full_df.to_numpy()

print (data)


# In[3]:


full_df.head(n=20)


# In[4]:


fig, ax = plt.subplots()

line1, = ax.plot(full_df['2500-64-branches'], label=" 64 bytes", linestyle='--')
line2, = ax.plot(full_df['2500-128-branches'], label=" 128 bytes", linestyle='--')
line3, = ax.plot(full_df['2500-512-branches'], label=" 512 bytes", linestyle='--')
line4, = ax.plot(full_df['2500-1400-branches'], label=" 1400 bytes", linestyle='--')
plt.title("2500 Mbps with various Data size")
plt.xlabel('Seconds')
plt.ylim(0,)
#plt.ylim(0, max(full_df['2500-64-branches'].max(), full_df['2500-128-branches'].max(), full_df['2500-512-branches'].max(),full_df['2500-1400-branches'].max()))
plt.ylabel('Normalized value of feature per time unit')
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')


fig, ax = plt.subplots()

line1, = ax.plot(full_df['5000-64-branches'], label=" 64 bytes", linestyle='--')
line2, = ax.plot(full_df['5000-128-branches'], label=" 128 bytes", linestyle='--')
line3, = ax.plot(full_df['5000-512-branches'], label=" 512 bytes", linestyle='--')
line4, = ax.plot(full_df['5000-1400-branches'], label=" 1400 bytes", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['5000-64-branches'].max(), full_df['5000-128-branches'].max(), full_df['5000-512-branches'].max(),full_df['5000-1400-branches'].max()))
plt.title("5000 Mbps with various Data size")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')


fig, ax = plt.subplots()

line1, = ax.plot(full_df['7500-64-branches'], label=" 64 bytes", linestyle='--')
line2, = ax.plot(full_df['7500-128-branches'], label=" 128 bytes", linestyle='--')
line3, = ax.plot(full_df['7500-512-branches'], label=" 512 bytes", linestyle='--')
line4, = ax.plot(full_df['7500-1400-branches'], label=" 1400 bytes", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['7500-64-branches'].max(), full_df['7500-128-branches'].max(), full_df['7500-512-branches'].max(),full_df['7500-1400-branches'].max()))
plt.title("7500 Mbps with various Data size")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')


fig, ax = plt.subplots()

line1, = ax.plot(full_df['10000-64-branches'], label=" 64 bytes", linestyle='--')
line2, = ax.plot(full_df['10000-128-branches'], label=" 128 bytes", linestyle='--')
line3, = ax.plot(full_df['10000-512-branches'], label=" 512 bytes", linestyle='--')
line4, = ax.plot(full_df['10000-1400-branches'], label=" 1400 bytes", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['10000-64-branches'].max(), full_df['10000-128-branches'].max(), full_df['10000-512-branches'].max(),full_df['10000-1400-branches'].max()))
plt.title("10000 Mbps with various Data size")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')


# In[5]:


fig, ax = plt.subplots()

line1, = ax.plot(full_df['2500-64-branches'], label="2500 Mbps", linestyle='--')
line2, = ax.plot(full_df['5000-64-branches'], label="5000 Mbps", linestyle='--')
line3, = ax.plot(full_df['7500-64-branches'], label="7500 Mbps", linestyle='--')
line4, = ax.plot(full_df['10000-64-branches'], label="10000 Mbps", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0,max(full_df['2500-64-branches'].max(),full_df['5000-64-branches'].max(),full_df['7500-64-branches'].max(),full_df['10000-64-branches'].max()))
plt.title("64 bytes with various Data rate")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')

################### 128

fig, ax = plt.subplots()

line1, = ax.plot(full_df['2500-128-branches'], label="2500 Mbps", linestyle='--')
line2, = ax.plot(full_df['5000-128-branches'], label="5000 Mbps", linestyle='--')
line3, = ax.plot(full_df['7500-128-branches'], label="7500 Mbps", linestyle='--')
line4, = ax.plot(full_df['10000-128-branches'], label="10000 Mbps", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['2500-128-branches'].max(), full_df['5000-128-branches'].max(), full_df['7500-128-branches'].max(),full_df['10000-128-branches'].max()))
plt.title("128 bytes with various Data rate")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')

################### 512
fig, ax = plt.subplots()

line1, = ax.plot(full_df['2500-512-branches'], label="2500 Mbps", linestyle='--')
line2, = ax.plot(full_df['5000-512-branches'], label="5000 Mbps", linestyle='--')
line3, = ax.plot(full_df['7500-512-branches'], label="7500 Mbps", linestyle='--')
line4, = ax.plot(full_df['10000-512-branches'], label="10000 Mbps", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['2500-512-branches'].max(), full_df['5000-512-branches'].max(), full_df['7500-512-branches'].max(),full_df['10000-512-branches'].max()))
plt.title("512 bytes with various Data rate")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')

################### 1400


fig, ax = plt.subplots()

line1, = ax.plot(full_df['2500-1400-branches'], label="2500 Mbps", linestyle='--')
line2, = ax.plot(full_df['5000-1400-branches'], label="5000 Mbps", linestyle='--')
line3, = ax.plot(full_df['7500-1400-branches'], label="7500 Mbps", linestyle='--')
line4, = ax.plot(full_df['10000-1400-branches'], label="10000 Mbps", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature per time unit')
plt.ylim(0,)
#plt.ylim(0, max(full_df['2500-1400-branches'].max(), full_df['5000-1400-branches'].max(), full_df['7500-1400-branches'].max(),full_df['10000-1400-branches'].max()))
plt.title("1400 bytes with various Data rate")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3, line4 ], loc='upper right')


# In[65]:


data_10000_rate = full_df.iloc[:,0:4]
minn=0
maxx=4
for x in range(0,155):
  minn= minn+20
  maxx= maxx+20
  data_10000_rate = pd.concat([data_10000_rate, full_df.iloc[:,minn:maxx]],axis=1)

data_10000_rate

data_2500_rate = full_df.iloc[:,4:8]
minn=4
maxx=8
for x in range(0,155):
  minn= minn+20
  maxx= maxx+20
  data_2500_rate = pd.concat([data_2500_rate, full_df.iloc[:,minn:maxx]],axis=1)

data_2500_rate

data_5000_rate = full_df.iloc[:,8:12]
minn=8
maxx=12
for x in range(0,155):
  minn= minn+20
  maxx= maxx+20
  data_5000_rate = pd.concat([data_5000_rate, full_df.iloc[:,minn:maxx]],axis=1)

data_5000_rate

data_500_rate = full_df.iloc[:,12:16]
minn=12
maxx=16
for x in range(0,155):
  minn= minn+20
  maxx= maxx+20
  data_500_rate = pd.concat([data_500_rate, full_df.iloc[:,minn:maxx]],axis=1)

data_500_rate

data_7500_rate = full_df.iloc[:,16:20]
minn=16
maxx=20
for x in range(0,155):
  minn= minn+20
  maxx= maxx+20
  data_7500_rate = pd.concat([data_7500_rate, full_df.iloc[:,minn:maxx]],axis=1)

data_7500_rate

# Classes definition base on rate

low_class = pd.concat([data_500_rate, data_2500_rate],axis=1)
med_class = data_5000_rate
hig_class = pd.concat([data_7500_rate, data_10000_rate],axis=1)


# In[6]:


zero_exp_low_class = low_class.iloc[0:15,:]
zero_exp_mid_class = med_class.iloc[0:15,:]
zero_exp_hig_class = hig_class.iloc[0:15,:]

#Let's consider the number of instructions
fig, ax = plt.subplots()
line1, = ax.plot(zero_exp_low_class['2500-512-instructions'], label="Low class 'instructions' at 512 packets size", linestyle='--')
line2, = ax.plot(zero_exp_mid_class['5000-512-instructions'], label="Mid class 'instructions' at 512 packets size", linestyle='--')
line3, = ax.plot(zero_exp_hig_class['10000-512-instructions'], label="Hig class 'instructions' at 512 packets size", linestyle='--')
plt.xlabel('Seconds')
plt.ylabel('Normalized value of feature (Instructions) per time unit')
plt.ylim(0,8000000000)
plt.title("512 bytes in each classes")
# Create a legend 
first_legend = ax.legend(handles=[line1,line2,line3 ], loc='upper right')


# In[14]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]
incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,3)
  #low class
  line1, = ax[0].plot(zero_exp_low_class["500-128-"+features[incr]], label="500-128", linestyle='--')
  line2, = ax[0].plot(zero_exp_low_class['500-1400-'+features[incr]], label="500-1400", linestyle='--')
  line3, = ax[0].plot(zero_exp_low_class['500-512-'+features[incr]], label="500-512", linestyle='--')
  line4, = ax[0].plot(zero_exp_low_class['500-64-'+features[incr]], label="500-64", linestyle='--')
  line5, = ax[0].plot(zero_exp_low_class['2500-128-'+features[incr]], label="2500-128", linestyle='--')
  line6, = ax[0].plot(zero_exp_low_class['2500-1400-'+features[incr]], label="2500-1400", linestyle='--')
  line7, = ax[0].plot(zero_exp_low_class['2500-512-'+features[incr]], label="2500-512", linestyle='--')
  line8, = ax[0].plot(zero_exp_low_class['2500-64-'+features[incr]], label="2500-64", linestyle='--')
  ax[0].set_xlabel('Seconds')
  ax[0].set_ylabel('low_class')
  ax[0].set_title(features[incr]) 
  # Create a legend 
  first_legend = ax[0].legend(handles=[line1,line2,line3,line4,line5,line6,line7,line8], loc='upper right')

  #middle class
  line1, = ax[1].plot(zero_exp_mid_class['5000-128-'+features[incr]], label="5000-128", linestyle='--')
  line2, = ax[1].plot(zero_exp_mid_class['5000-1400-'+features[incr]], label="5000-1400", linestyle='--')
  line3, = ax[1].plot(zero_exp_mid_class['5000-512-'+features[incr]], label="5000-512", linestyle='--')
  line4, = ax[1].plot(zero_exp_mid_class['5000-64-'+features[incr]], label="5000-64", linestyle='--')
  ax[1].set_xlabel('Seconds')
  ax[1].set_ylabel('mid_class')
  ax[1].set_title(features[incr]) 
  # Create a legend 
  first_legend = ax[1].legend(handles=[line1,line2,line3,line4], loc='upper right')

  #hig class
  line1, = ax[2].plot(zero_exp_hig_class['7500-128-'+features[incr]], label="7500-128", linestyle='--')
  line2, = ax[2].plot(zero_exp_hig_class['7500-1400-'+features[incr]], label="7500-1400", linestyle='--')
  line3, = ax[2].plot(zero_exp_hig_class['7500-512-'+features[incr]], label="7500-512", linestyle='--')
  line4, = ax[2].plot(zero_exp_hig_class['7500-64-'+features[incr]], label="7500-64", linestyle='--')
  line5, = ax[2].plot(zero_exp_hig_class['10000-128-'+features[incr]], label="10000-128", linestyle='--')
  line6, = ax[2].plot(zero_exp_hig_class['10000-1400-'+features[incr]], label="10000-1400", linestyle='--')
  line7, = ax[2].plot(zero_exp_hig_class['10000-512-'+features[incr]], label="10000-512", linestyle='--')
  line8, = ax[2].plot(zero_exp_hig_class['10000-64-'+features[incr]], label="10000-64", linestyle='--')
  ax[2].set_xlabel('Seconds')
  ax[2].set_ylabel('hig_class')
  ax[2].set_title(features[incr]) 
  # Create a legend 
  first_legend = ax[2].legend(handles=[line1,line2,line3,line4,line5,line6,line7,line8], loc='upper right')

  fig.set_size_inches(50, 18)
  fig.suptitle('All features with variate data_size (128,1400,512,64 data_size) in all classes of EXP_ZERO', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1


# In[7]:


#General dataframe containing average and stddev of full Zero_exp

avg_stddev_zero_exp = {'NameofExp':[], 'Avg':[],'Stddev':[]}
# creating a dataframe from dictionary
avg_stddev_zero_exp_df = pd.DataFrame(avg_stddev_zero_exp)

features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]
incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,3)
  #low class
  #dataframe for low_class
  avg_stddev_zero_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  avg_stddev_zero_exp_low_df = pd.DataFrame(avg_stddev_zero_exp_low)

  line1, = ax[0].plot(zero_exp_low_class["500-128-"+features[incr]], label="500-128", linestyle='--')
  new_element = {'NameofExp':"500-128" , 'Avg': zero_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-128-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line2, = ax[0].plot(zero_exp_low_class['500-1400-'+features[incr]], label="500-1400", linestyle='--')
  new_element = {'NameofExp':"500-1400" , 'Avg': zero_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-1400-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line3, = ax[0].plot(zero_exp_low_class['500-512-'+features[incr]], label="500-512", linestyle='--')
  new_element = {'NameofExp':"500-512" , 'Avg': zero_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-512-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line4, = ax[0].plot(zero_exp_low_class['500-64-'+features[incr]], label="500-64", linestyle='--')
  new_element = {'NameofExp':"500-64" , 'Avg': zero_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-64-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line5, = ax[0].plot(zero_exp_low_class['2500-128-'+features[incr]], label="2500-128", linestyle='--')
  new_element = {'NameofExp':"2500-128" , 'Avg': zero_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-128-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line6, = ax[0].plot(zero_exp_low_class['2500-1400-'+features[incr]], label="2500-1400", linestyle='--')
  new_element = {'NameofExp':"2500-1400" , 'Avg': zero_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-1400-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line7, = ax[0].plot(zero_exp_low_class['2500-512-'+features[incr]], label="2500-512", linestyle='--')
  new_element = {'NameofExp':"2500-512" , 'Avg': zero_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-512-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  line8, = ax[0].plot(zero_exp_low_class['2500-64-'+features[incr]], label="2500-64", linestyle='--')
  new_element = {'NameofExp':"2500-64" , 'Avg': zero_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-64-"+features[incr]].std()}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  lineavg, = ax[0].plot(avg_stddev_zero_exp_low_df['Avg'].iloc[-8:-1], label="avg_low_class", linestyle='solid',linewidth=4.0)
  linestd, = ax[0].plot(avg_stddev_zero_exp_low_df['Stddev'].iloc[-8:-1], label="std_low_class", linestyle='solid',linewidth=4.0)
  ax[0].set_xlabel('Seconds')
  ax[0].set_ylabel('low_class')
  ax[0].set_title(features[incr])
  # avg and stddev saving for global exp 
  avg_stddev_zero_exp_df.append(avg_stddev_zero_exp_low_df)

  # Adding Twin Axes
  ax2 = ax[0].twinx() 
  ax2.set_ylabel('std_axis', color = 'blue')
  ax2.set_ylim(0,1)
  # Create a legend 
  first_legend = ax[0].legend(handles=[line1,line2,line3,line4,line5,line6,line7,line8,lineavg,linestd], loc='upper right')

  #middle class
  #for mid_class
  avg_stddev_zero_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  avg_stddev_zero_exp_mid_df = pd.DataFrame(avg_stddev_zero_exp_mid)

  line1, = ax[1].plot(zero_exp_mid_class['5000-128-'+features[incr]], label="5000-128", linestyle='--')
  new_element = {'NameofExp':"5000-128" , 'Avg': zero_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-128-"+features[incr]].std()}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  line2, = ax[1].plot(zero_exp_mid_class['5000-1400-'+features[incr]], label="5000-1400", linestyle='--')
  new_element = {'NameofExp':"5000-1400" , 'Avg': zero_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-1400-"+features[incr]].std()}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  line3, = ax[1].plot(zero_exp_mid_class['5000-512-'+features[incr]], label="5000-512", linestyle='--')
  new_element = {'NameofExp':"5000-512" , 'Avg': zero_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-512-"+features[incr]].std()}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  line4, = ax[1].plot(zero_exp_mid_class['5000-64-'+features[incr]], label="5000-64", linestyle='--')
  new_element = {'NameofExp':"5000-64" , 'Avg': zero_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-64-"+features[incr]].std()}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  lineavg, = ax[1].plot(avg_stddev_zero_exp_mid_df['Avg'].iloc[-8:-1], label="avg_mid_class", linestyle='solid',linewidth=4.0)
  linestd, = ax[1].plot(avg_stddev_zero_exp_mid_df['Stddev'].iloc[-8:-1], label="std_mid_class", linestyle='solid',linewidth=4.0)
  ax[1].set_xlabel('Seconds')
  ax[1].set_ylabel('mid_class')
  ax[1].set_title(features[incr])
  # avg and stddev saving for global exp 
  avg_stddev_zero_exp_df.append(avg_stddev_zero_exp_mid_df)

  # Adding Twin Axes
  ax2 = ax[1].twinx() 
  ax2.set_ylabel('std_axis', color = 'blue')
  ax2.set_ylim(0,1)
  # Create a legend 
  first_legend = ax[1].legend(handles=[line1,line2,line3,line4,lineavg,linestd], loc='upper right')

  #hig class
  #for hig_class
  avg_stddev_zero_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[]}
  # creating a dataframe from dictionary
  avg_stddev_zero_exp_hig_df = pd.DataFrame(avg_stddev_zero_exp_hig)

  line1, = ax[2].plot(zero_exp_hig_class['7500-128-'+features[incr]], label="7500-128", linestyle='--')
  new_element = {'NameofExp':"7500-128" , 'Avg': zero_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-128-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line2, = ax[2].plot(zero_exp_hig_class['7500-1400-'+features[incr]], label="7500-1400", linestyle='--')
  new_element = {'NameofExp':"7500-1400" , 'Avg': zero_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-1400-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line3, = ax[2].plot(zero_exp_hig_class['7500-512-'+features[incr]], label="7500-512", linestyle='--')
  new_element = {'NameofExp':"7500-512" , 'Avg': zero_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-512-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line4, = ax[2].plot(zero_exp_hig_class['7500-64-'+features[incr]], label="7500-64", linestyle='--')
  new_element = {'NameofExp':"7500-64" , 'Avg': zero_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-64-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line5, = ax[2].plot(zero_exp_hig_class['10000-128-'+features[incr]], label="10000-128", linestyle='--')
  new_element = {'NameofExp':"10000-128" , 'Avg': zero_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-128-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line6, = ax[2].plot(zero_exp_hig_class['10000-1400-'+features[incr]], label="10000-1400", linestyle='--')
  new_element = {'NameofExp':"10000-1400" , 'Avg': zero_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-1400-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line7, = ax[2].plot(zero_exp_hig_class['10000-512-'+features[incr]], label="10000-512", linestyle='--')
  new_element = {'NameofExp':"10000-512" , 'Avg': zero_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-512-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  line8, = ax[2].plot(zero_exp_hig_class['10000-64-'+features[incr]], label="10000-64", linestyle='--')
  new_element = {'NameofExp':"10000-64" , 'Avg': zero_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-64-"+features[incr]].std()}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  lineavg, = ax[2].plot(avg_stddev_zero_exp_hig_df['Avg'].iloc[-8:-1], label="avg_hig_class", linestyle='solid',linewidth=4.0)
  linestd, = ax[2].plot(avg_stddev_zero_exp_hig_df['Stddev'].iloc[-8:-1], label="std_hig_class", linestyle='solid',linewidth=4.0)
  ax[2].set_xlabel('Seconds')
  ax[2].set_ylabel('hig_class')
  ax[2].set_title(features[incr])
  # avg and stddev saving for global exp 
  avg_stddev_zero_exp_df.append(avg_stddev_zero_exp_hig_df)

  # Adding Twin Axes
  ax2 = ax[2].twinx() 
  ax2.set_ylabel('std_axis', color = 'blue')
  ax2.set_ylim(0,1)
  # Create a legend 
  first_legend = ax[2].legend(handles=[line1,line2,line3,line4,line5,line6,line7,line8,lineavg,linestd], loc='upper right')

  fig.set_size_inches(50, 18)
  fig.suptitle('All features with variate data_size (128,1400,512,64 data_size) in all classes of EXP_ZERO', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1


# In[8]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

#dataframe for low_class
avg_stddev_zero_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_zero_exp_low_df = pd.DataFrame(avg_stddev_zero_exp_low)

#dataframe for mid_class
avg_stddev_zero_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_zero_exp_mid_df = pd.DataFrame(avg_stddev_zero_exp_mid)

#dataframe for hig_class
avg_stddev_zero_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_zero_exp_hig_df = pd.DataFrame(avg_stddev_zero_exp_hig)

#dataframe for ZERO_EXP_LOW
zero_exp_low = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
zero_exp_low_df = pd.DataFrame(zero_exp_low)

#dataframe for ZERO_EXP_MID
zero_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
zero_exp_mid_df = pd.DataFrame(zero_exp_mid)

#dataframe for ZERO_EXP_HIG
zero_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
zero_exp_hig_df = pd.DataFrame(zero_exp_hig)


incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(zero_exp_low_class["500-128-"+features[incr]].mean(),zero_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-128" , 'Avg': zero_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-128', xy =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-1400-"+features[incr]].mean(),zero_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': zero_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': zero_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': zero_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': zero_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-1400-"+features[incr]].mean(),  zero_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': zero_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': zero_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': zero_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_low_df = avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("FULL SHAPE")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_zero_exp_low_df["Avg"]), 'Stddev': avg_stddev_zero_exp_low_df["Stddev"].min()
                ,'CI_lower': avg_stddev_zero_exp_low_df["CI_lower"].min(),'CI_upper': avg_stddev_zero_exp_low_df["CI_upper"].max()}
  zero_exp_low_df = zero_exp_low_df.append(new_element, ignore_index = True)


 

  #middle class
  ax[1].scatter(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': zero_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': zero_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))  
  new_element = {'NameofExp':"5000-512" , 'Avg': zero_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': zero_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_mid_df = avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_zero_exp_mid_df["Avg"]), 'Stddev': avg_stddev_zero_exp_mid_df["Stddev"].min()
                ,'CI_lower': avg_stddev_zero_exp_mid_df["CI_lower"].min(),'CI_upper': avg_stddev_zero_exp_mid_df["CI_upper"].max()}
  zero_exp_mid_df = zero_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': zero_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': zero_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': zero_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': zero_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': zero_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': zero_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': zero_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': zero_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_zero_exp_hig_df = avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_zero_exp_hig_df["Avg"]), 'Stddev': avg_stddev_zero_exp_hig_df["Stddev"].min()
                ,'CI_lower': avg_stddev_zero_exp_hig_df["CI_lower"].min(),'CI_upper': avg_stddev_zero_exp_hig_df["CI_upper"].max()}
  zero_exp_hig_df = zero_exp_hig_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(avg_stddev_zero_exp_low_df["Avg"].tail(8), avg_stddev_zero_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(avg_stddev_zero_exp_mid_df["Avg"].tail(4), avg_stddev_zero_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(avg_stddev_zero_exp_hig_df["Avg"].tail(8), avg_stddev_zero_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')
  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1

#Saving zero_exp_mid_df into csv
zero_exp_low_df.to_csv("zero_exp_low.csv")
zero_exp_mid_df.to_csv("zero_exp_mid.csv")
zero_exp_hig_df.to_csv("zero_exp_hig.csv")


# In[9]:


# Principal components for Zero_exp

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

#dataframe for low_class
PC_avg_stddev_zero_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_zero_exp_low_df = pd.DataFrame(PC_avg_stddev_zero_exp_low)

#dataframe for mid_class
PC_avg_stddev_zero_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_zero_exp_mid_df = pd.DataFrame(PC_avg_stddev_zero_exp_mid)

#dataframe for hig_class
PC_avg_stddev_zero_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_zero_exp_hig_df = pd.DataFrame(PC_avg_stddev_zero_exp_hig)


#dataframe for ZERO_EXP_LOW
PC_zero_exp_low = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_zero_exp_low_df = pd.DataFrame(PC_zero_exp_low)

#dataframe for ZERO_EXP_MID
PC_zero_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_zero_exp_mid_df = pd.DataFrame(PC_zero_exp_mid)

#dataframe for ZERO_EXP_HIG
PC_zero_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
PC_zero_exp_hig_df = pd.DataFrame(PC_zero_exp_hig)

#dataframe for ZERO_EXP
PC_zero_exp = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
PC_zero_exp_df = pd.DataFrame(PC_zero_exp)


incr =0
for incr in range(0,7):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(zero_exp_low_class["500-128-"+features[incr]].mean(),zero_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-128" , 'Avg': zero_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-128', xy =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-1400-"+features[incr]].mean(),zero_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': zero_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': zero_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': zero_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': zero_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-1400-"+features[incr]].mean(),  zero_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': zero_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': zero_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= zero_exp_low_class["2500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': zero_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': zero_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_low_df = PC_avg_stddev_zero_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("FULL SHAPE")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_zero_exp_low_df["Avg"]), 'Stddev': PC_avg_stddev_zero_exp_low_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_zero_exp_low_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_zero_exp_low_df["CI_upper"].max()}
  PC_zero_exp_low_df = PC_zero_exp_low_df.append(new_element, ignore_index = True)


  #middle class
  ax[1].scatter(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': zero_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_mid_df = PC_avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': zero_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_mid_df = PC_avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))  
  new_element = {'NameofExp':"5000-512" , 'Avg': zero_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_mid_df = PC_avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= zero_exp_mid_class["5000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': zero_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': zero_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_mid_df = PC_avg_stddev_zero_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_zero_exp_mid_df["Avg"]), 'Stddev': PC_avg_stddev_zero_exp_mid_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_zero_exp_mid_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_zero_exp_mid_df["CI_upper"].max()}
  PC_zero_exp_mid_df = PC_zero_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': zero_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': zero_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': zero_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["7500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': zero_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': zero_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': zero_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': zero_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= zero_exp_hig_class["10000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': zero_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': zero_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_zero_exp_hig_df = PC_avg_stddev_zero_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_zero_exp_hig_df["Avg"]), 'Stddev': PC_avg_stddev_zero_exp_hig_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_zero_exp_hig_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_zero_exp_hig_df["CI_upper"].max()}
  PC_zero_exp_hig_df = PC_zero_exp_hig_df.append(new_element, ignore_index = True)


  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': min(statistics.median(PC_avg_stddev_zero_exp_low_df["Avg"]),statistics.median(PC_avg_stddev_zero_exp_mid_df["Avg"]),statistics.median(PC_avg_stddev_zero_exp_hig_df["Avg"])), 
                 'Stddev': min(PC_avg_stddev_zero_exp_low_df["Stddev"].min(),PC_avg_stddev_zero_exp_mid_df["Stddev"].min(),PC_avg_stddev_zero_exp_hig_df["Stddev"].min())
                ,'CI_lower': min(PC_avg_stddev_zero_exp_low_df["CI_lower"].min(),PC_avg_stddev_zero_exp_mid_df["CI_lower"].min(),PC_avg_stddev_zero_exp_hig_df["CI_lower"].min()),
                 'CI_upper': max(PC_avg_stddev_zero_exp_low_df["CI_upper"].max(),PC_avg_stddev_zero_exp_mid_df["CI_upper"].max(),PC_avg_stddev_zero_exp_hig_df["CI_upper"].max())}
  PC_zero_exp_df = PC_zero_exp_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(PC_avg_stddev_zero_exp_low_df["Avg"].tail(8), PC_avg_stddev_zero_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(PC_avg_stddev_zero_exp_mid_df["Avg"].tail(4), PC_avg_stddev_zero_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(PC_avg_stddev_zero_exp_hig_df["Avg"].tail(8), PC_avg_stddev_zero_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')
  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1

#Saving zero_exp_df into csv
PC_zero_exp_df.to_csv("PC_zero_exp.csv")


# In[12]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,1)
  #low class
  ax.scatter(zero_exp_low_class["500-128-"+features[incr]].mean(),zero_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  ax.annotate('500-128', xy =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-128-"+features[incr]].mean(), zero_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax.set_xlabel("Avg")
  ax.set_ylabel('Stddev')

  ax.scatter(zero_exp_low_class["500-1400-"+features[incr]].mean(),zero_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  ax.annotate('500-1400', xy =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-1400-"+features[incr]].mean(), zero_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )

  ax.scatter(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  ax.annotate('500-512', xy =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-512-"+features[incr]].mean(), zero_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  ax.annotate('500-64', xy =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["500-64-"+features[incr]].mean(), zero_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )


  ax.scatter(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  ax.annotate('2500-128', xy =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-128-"+features[incr]].mean(), zero_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )

  ax.scatter(zero_exp_low_class["2500-1400-"+features[incr]].mean(),  zero_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  ax.annotate('2500-1400', xy =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-1400-"+features[incr]].mean(), zero_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  ax.annotate('2500-512', xy =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-512-"+features[incr]].mean(), zero_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )

  ax.scatter(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  ax.annotate('2500-64', xy =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(zero_exp_low_class["2500-64-"+features[incr]].mean(), zero_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  
  ax.set_xlabel("Avg")
  ax.set_ylabel('Stddev')
  ax.set_title("FULL SHAPE")

 
 

  #middle class
  ax.scatter(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  ax.annotate('5000-128', xy =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-128-"+features[incr]].mean(), zero_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  ax.annotate('5000-1400', xy =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-1400-"+features[incr]].mean(), zero_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  
  
  ax.scatter(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  ax.annotate('5000-512', xy =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-512-"+features[incr]].mean(), zero_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  ax.annotate('5000-64', xy =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(zero_exp_mid_class["5000-64-"+features[incr]].mean(), zero_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  

 
  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax.scatter(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  ax.annotate('7500-128', xy =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-128-"+features[incr]].mean(), zero_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  
  

  ax.scatter(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  ax.annotate('7500-1400', xy =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-1400-"+features[incr]].mean(), zero_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  ax.annotate('7500-512', xy =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-512-"+features[incr]].mean(), zero_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  
  

  ax.scatter(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  ax.annotate('7500-64', xy =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["7500-64-"+features[incr]].mean(), zero_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  
  

  ax.scatter(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  ax.annotate('10000-128', xy =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-128-"+features[incr]].mean(), zero_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  
  
  ax.scatter(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  ax.annotate('10000-1400', xy =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-1400-"+features[incr]].mean(), zero_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  ax.annotate('10000-512', xy =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-512-"+features[incr]].mean(), zero_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  

  ax.scatter(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  ax.annotate('10000-64', xy =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(zero_exp_hig_class["10000-64-"+features[incr]].mean(), zero_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  
  
  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')
  

  fig.set_size_inches(12, 7)
  fig.suptitle(features[incr]+' ALL-CLASSES classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1


# In[66]:


interesting_exp_low_class = low_class.iloc[19:30,:]
interesting_exp_mid_class = med_class.iloc[19:30,:]
interesting_exp_hig_class = hig_class.iloc[19:30,:]

max_rate_exp = pd.concat([interesting_exp_low_class, interesting_exp_mid_class,interesting_exp_hig_class],axis=1)

#max_rate_{rate}_{packets_size}_bytes

#max_rate_500_128_bytes
max_rate_500_128_bytes =  interesting_exp_low_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  max_rate_500_128_bytes =  pd.concat([max_rate_500_128_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_500_128_bytes

#max_rate_500_1400_bytes
max_rate_500_1400_bytes = interesting_exp_low_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  max_rate_500_1400_bytes =  pd.concat([max_rate_500_1400_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_500_1400_bytes


#max_rate_500_512_bytes
max_rate_500_512_bytes = interesting_exp_low_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  max_rate_500_512_bytes =  pd.concat([max_rate_500_512_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_500_512_bytes

#max_rate_500_64_bytes
max_rate_500_64_bytes = interesting_exp_low_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  max_rate_500_64_bytes =  pd.concat([max_rate_500_64_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_500_64_bytes


#max_rate_2500_128_bytes
max_rate_2500_128_bytes = interesting_exp_low_class.iloc[:,124]
incr=124
for x in range(0,30):
  incr= incr+4
  max_rate_2500_128_bytes =  pd.concat([max_rate_2500_128_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_2500_128_bytes

#max_rate_2500_1400_bytes
max_rate_2500_1400_bytes = interesting_exp_low_class.iloc[:,125]
incr=125
for x in range(0,30):
  incr= incr+4
  max_rate_2500_1400_bytes =  pd.concat([max_rate_2500_1400_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_2500_1400_bytes

#max_rate_2500_512_bytes
max_rate_2500_512_bytes = interesting_exp_low_class.iloc[:,126]
incr=126
for x in range(0,30):
  incr= incr+4
  max_rate_2500_512_bytes =  pd.concat([max_rate_2500_512_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_2500_512_bytes

#max_rate_2500_64_bytes
max_rate_2500_64_bytes = interesting_exp_low_class.iloc[:,127]
incr=127
for x in range(0,30):
  incr= incr+4
  max_rate_2500_64_bytes =  pd.concat([max_rate_2500_64_bytes, interesting_exp_low_class.iloc[:,incr]],axis=1)

max_rate_2500_64_bytes


#max_rate_5000_128_bytes
max_rate_5000_128_bytes =  interesting_exp_mid_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  max_rate_5000_128_bytes =  pd.concat([max_rate_5000_128_bytes, interesting_exp_mid_class.iloc[:,incr]],axis=1)

max_rate_5000_128_bytes

#max_rate_5000_1400_bytes
max_rate_5000_1400_bytes = interesting_exp_mid_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  max_rate_5000_1400_bytes =  pd.concat([max_rate_5000_1400_bytes, interesting_exp_mid_class.iloc[:,incr]],axis=1)

max_rate_5000_1400_bytes


#max_rate_5000_512_bytes
max_rate_5000_512_bytes = interesting_exp_mid_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  max_rate_5000_512_bytes =  pd.concat([max_rate_5000_512_bytes, interesting_exp_mid_class.iloc[:,incr]],axis=1)

max_rate_5000_512_bytes

#max_rate_5000_64_bytes
max_rate_5000_64_bytes = interesting_exp_mid_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  max_rate_5000_64_bytes =  pd.concat([max_rate_5000_64_bytes, interesting_exp_mid_class.iloc[:,incr]],axis=1)

max_rate_5000_64_bytes


#max_rate_7500_128_bytes
max_rate_7500_128_bytes =  interesting_exp_hig_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  max_rate_7500_128_bytes =  pd.concat([max_rate_7500_128_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_7500_128_bytes

#max_rate_7500_1400_bytes
max_rate_7500_1400_bytes = interesting_exp_hig_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  max_rate_7500_1400_bytes =  pd.concat([max_rate_7500_1400_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_7500_1400_bytes


#max_rate_7500_512_bytes
max_rate_7500_512_bytes = interesting_exp_hig_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  max_rate_7500_512_bytes =  pd.concat([max_rate_7500_512_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_7500_512_bytes

#max_rate_7500_64_bytes
max_rate_7500_64_bytes = interesting_exp_hig_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  max_rate_7500_64_bytes =  pd.concat([max_rate_7500_64_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_7500_64_bytes


#max_rate_10000_128_bytes
max_rate_10000_128_bytes = interesting_exp_hig_class.iloc[:,124]
incr=124
for x in range(0,30):
  incr= incr+4
  max_rate_10000_128_bytes =  pd.concat([max_rate_10000_128_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_10000_128_bytes

#max_rate_10000_1400_bytes
max_rate_10000_1400_bytes = interesting_exp_hig_class.iloc[:,125]
incr=125
for x in range(0,30):
  incr= incr+4
  max_rate_10000_1400_bytes =  pd.concat([max_rate_10000_1400_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_10000_1400_bytes

#max_rate_10000_512_bytes
max_rate_10000_512_bytes = interesting_exp_hig_class.iloc[:,126]
incr=126
for x in range(0,30):
  incr= incr+4
  max_rate_10000_512_bytes =  pd.concat([max_rate_10000_512_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

max_rate_10000_512_bytes

#max_rate_10000_64_bytes
max_rate_10000_64_bytes = interesting_exp_hig_class.iloc[:,127]
incr=127
for x in range(0,30):
  incr= incr+4
  max_rate_10000_64_bytes =  pd.concat([max_rate_10000_64_bytes, interesting_exp_hig_class.iloc[:,incr]],axis=1)

#max_rate_10000_64_bytes



final_exp= pd.concat([interesting_exp_low_class,interesting_exp_mid_class,interesting_exp_hig_class ],axis=1)


# In[22]:


#Full part in low class

#For 500 data_rate with differents data_size
fig, ax = plt.subplots(4,30)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]
incr =0
for incr in range(0,30):
  ax[0,incr].plot(max_rate_500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(max_rate_500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(max_rate_500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(max_rate_500_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('500 date_rate at 128,1400,512,64 data_size', fontsize=16)


#For 2500 data_rate with differents data_size
fig, ax = plt.subplots(4,30)
incr =0
for incr in range(0,30):
  ax[0,incr].plot(max_rate_2500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(max_rate_2500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(max_rate_2500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(max_rate_2500_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('2500 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[23]:


#For 7500 data_rate with differents data_size
fig, ax = plt.subplots(4,31)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  ax[0,incr].plot(max_rate_7500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(max_rate_7500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(max_rate_7500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(max_rate_7500_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('7500 date_rate at 128,1400,512,64 data_size', fontsize=16)


#For 10000 data_rate with differents data_size
fig, ax = plt.subplots(4,31)
incr =0
for incr in range(0,30):
  ax[0,incr].plot(max_rate_10000_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(max_rate_10000_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(max_rate_10000_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(max_rate_10000_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('10000 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[24]:


#For 5000 data_rate with differents data_size
fig, ax = plt.subplots(4,31)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  ax[0,incr].plot(max_rate_5000_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(max_rate_5000_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(max_rate_5000_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(max_rate_5000_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('7500 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[67]:


full_exp_low_class = low_class.iloc[19:30,:]
full_exp_mid_class = med_class.iloc[19:30,:]
full_exp_hig_class = hig_class.iloc[19:30,:]


# In[26]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]
incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(5,4)
  #low class

  ax[0,0].hist(full_exp_low_class["500-128-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["500-128-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["500-128-"+features[incr]].min(),full_exp_low_class["500-128-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[0,0].plot(x, p, 'k', linewidth=2,color="red")
  ax[0,0].set_xlabel("500-128-"+features[incr])
  ax[0,0].set_ylabel('Count')
  ax[0,0].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[0,0].legend(handles=[line2], loc='upper right')

  ax[0,1].hist(full_exp_low_class["500-1400-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["500-1400-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["500-1400-"+features[incr]].min(),full_exp_low_class["500-1400-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[0,1].plot(x, p, 'k', linewidth=2,color="red")
  ax[0,1].set_xlabel("500-1400-"+features[incr])
  ax[0,1].set_ylabel('Count')
  ax[0,1].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[0,1].legend(handles=[line2], loc='upper right')

  ax[0,2].hist(full_exp_low_class["500-512-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["500-512-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["500-512-"+features[incr]].min(),full_exp_low_class["500-512-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[0,2].plot(x, p, 'k', linewidth=2,color="red")
  ax[0,2].set_xlabel("500-512-"+features[incr])
  ax[0,2].set_ylabel('Count')
  ax[0,2].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[0,2].legend(handles=[line2], loc='upper right')

  ax[0,3].hist(full_exp_low_class["500-64-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["500-64-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["500-64-"+features[incr]].min(),full_exp_low_class["500-64-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[0,3].plot(x, p, 'k', linewidth=2,color="red")
  
  ax[0,3].set_xlabel("500-64-"+features[incr])
  ax[0,3].set_ylabel('Count')
  ax[0,3].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[0,3].legend(handles=[line2], loc='upper right')

  ax[1,0].hist(full_exp_low_class["2500-128-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["2500-128-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["2500-128-"+features[incr]].min(),full_exp_low_class["2500-128-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[1,0].plot(x, p, 'k', linewidth=2,color="red")
  ax[1,0].set_xlabel("2500-128-"+features[incr])
  ax[1,0].set_ylabel('Count')
  ax[1,0].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[1,0].legend(handles=[line2], loc='upper right')

  ax[1,1].hist(full_exp_low_class["2500-1400-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["2500-1400-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["2500-1400-"+features[incr]].min(),full_exp_low_class["2500-1400-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[1,1].plot(x, p, 'k', linewidth=2,color="red")

  #line2, = ax[1,1].plot(full_exp_low_class["2500-1400-"+features[incr]].mean(), label="mean", linestyle='solid')
  ax[1,1].set_xlabel("2500-1400-"+features[incr])
  ax[1,1].set_ylabel('Count')
  ax[1,1].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[1,1].legend(handles=[line2], loc='upper right')

  ax[1,2].hist(full_exp_low_class["2500-512-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["2500-512-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["2500-512-"+features[incr]].min(),full_exp_low_class["2500-512-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[1,2].plot(x, p, 'k', linewidth=2,color="red")

  ax[1,2].set_xlabel("2500-512-"+features[incr])
  ax[1,2].set_ylabel('Count')
  ax[1,2].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[1,2].legend(handles=[line2], loc='upper right')

  ax[1,3].hist(full_exp_low_class["2500-64-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_low_class["2500-64-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_low_class["2500-64-"+features[incr]].min(),full_exp_low_class["2500-64-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[1,3].plot(x, p, 'k', linewidth=2,color="red")

  ax[1,3].set_xlabel("2500-64-"+features[incr])
  ax[1,3].set_ylabel('Count')
  ax[1,3].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[1,3].legend(handles=[line2], loc='upper right')

  #middle class
  ax[2,0].hist(full_exp_mid_class["5000-128-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_mid_class["5000-128-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_mid_class["5000-128-"+features[incr]].min(),full_exp_mid_class["5000-128-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[2,0].plot(x, p, 'k', linewidth=2,color="red")

  ax[2,0].set_xlabel("5000-128-"+features[incr])
  ax[2,0].set_ylabel('Count')
  ax[2,0].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[2,0].legend(handles=[line2], loc='upper right')

  ax[2,1].hist(full_exp_mid_class["5000-1400-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_mid_class["5000-1400-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_mid_class["5000-1400-"+features[incr]].min(),full_exp_mid_class["5000-1400-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[2,1].plot(x, p, 'k', linewidth=2,color="red")
  
  ax[2,1].set_xlabel("5000-1400-"+features[incr])
  ax[2,1].set_ylabel('Count')
  ax[2,1].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[2,1].legend(handles=[line2], loc='upper right')
  
  ax[2,2].hist(full_exp_mid_class["5000-512-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_mid_class["5000-512-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_mid_class["5000-512-"+features[incr]].min(),full_exp_mid_class["5000-512-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[2,2].plot(x, p, 'k', linewidth=2,color="red")

  ax[2,2].set_xlabel("5000-512-"+features[incr])
  ax[2,2].set_ylabel('Count')
  ax[2,2].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[2,2].legend(handles=[line2], loc='upper right')

  ax[2,3].hist(full_exp_mid_class["5000-64-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_mid_class["5000-64-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_mid_class["5000-64-"+features[incr]].min(),full_exp_mid_class["5000-64-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[2,3].plot(x, p, 'k', linewidth=2,color="red")

  ax[2,3].set_xlabel("5000-64-"+features[incr])
  ax[2,3].set_ylabel('Count')
  ax[2,3].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[2,3].legend(handles=[line2], loc='upper right')
  
  #hig class

  ax[3,0].hist(full_exp_hig_class["7500-128-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["7500-128-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["7500-128-"+features[incr]].min(),full_exp_hig_class["7500-128-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  
  plt.plot(x, p, 'k', linewidth=2,color="red")

  line2, = ax[3,0].plot(x, p, 'k', linewidth=2,color="red")

  ax[3,0].set_xlabel("7500-128-"+features[incr])
  ax[3,0].set_ylabel('Count')
  ax[3,0].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[3,0].legend(handles=[line2], loc='upper right')

  ax[3,1].hist(full_exp_hig_class["7500-1400-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["7500-1400-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["7500-1400-"+features[incr]].min(),full_exp_hig_class["7500-1400-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[3,1].plot(x, p, 'k', linewidth=2,color="red")
  
  ax[3,1].set_xlabel("7500-1400-"+features[incr])
  ax[3,1].set_ylabel('Count')
  ax[3,1].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[3,1].legend(handles=[line2], loc='upper right')

  ax[3,2].hist(full_exp_hig_class["7500-512-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["7500-512-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["7500-512-"+features[incr]].min(),full_exp_hig_class["7500-512-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[3,2].plot(x, p, 'k', linewidth=2,color="red")

  #line2, = ax[3,2].plot(full_exp_hig_class["7500-512-"+features[incr]].mean(), label="mean", linestyle='solid')
  ax[3,2].set_xlabel("7500-512-"+features[incr])
  ax[3,2].set_ylabel('Count')
  ax[3,2].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[3,2].legend(handles=[line2], loc='upper right')

  ax[3,3].hist(full_exp_hig_class["7500-64-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["7500-64-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["7500-64-"+features[incr]].min(),full_exp_hig_class["7500-64-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[3,3].plot(x, p, 'k', linewidth=2,color="red")

  ax[3,3].set_xlabel("7500-64-"+features[incr])
  ax[3,3].set_ylabel('Count')
  ax[3,3].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[3,3].legend(handles=[line2], loc='upper right')

  ax[4,0].hist(full_exp_hig_class["10000-128-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["10000-128-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["10000-128-"+features[incr]].min(),full_exp_hig_class["10000-128-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[4,0].plot(x, p, 'k', linewidth=2,color="red")
  ax[4,0].set_xlabel("10000-128-"+features[incr])
  ax[4,0].set_ylabel('Count')
  ax[4,0].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[4,0].legend(handles=[line2], loc='upper right')

  ax[4,1].hist(full_exp_hig_class["10000-1400-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["10000-1400-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["10000-1400-"+features[incr]].min(),full_exp_hig_class["10000-1400-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[4,1].plot(x, p, 'k', linewidth=2,color="red")

  #line2, = ax[4,1].plot(full_exp_hig_class["10000-1400-"+features[incr]].mean(), label="mean", linestyle='solid')
  ax[4,1].set_xlabel("10000-1400-"+features[incr])
  ax[4,1].set_ylabel('Count')
  ax[4,1].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[4,1].legend(handles=[line2], loc='upper right')

  ax[4,2].hist(full_exp_hig_class["10000-512-"+features[incr]],bins=20, density=True)

  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["10000-512-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["10000-512-"+features[incr]].min(),full_exp_hig_class["10000-512-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[4,2].plot(x, p, 'k', linewidth=2,color="red")
  
  ax[4,2].set_xlabel("10000-512-"+features[incr])
  ax[4,2].set_ylabel('Count')
  ax[4,2].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[4,2].legend(handles=[line2], loc='upper right')

  ax[4,3].hist(full_exp_hig_class["10000-64-"+features[incr]],bins=20, density=True)
  #Fit a normal distribution to the data:
  # mean and standard deviation
  mu, std = norm.fit(full_exp_hig_class["10000-64-"+features[incr]]) 
  # Plot the PDF.
  xmin, xmax = (full_exp_hig_class["10000-64-"+features[incr]].min(),full_exp_hig_class["10000-64-"+features[incr]].max())
  x = np.linspace(xmin, xmax)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2,color="red")
  line2, = ax[4,3].plot(x, p, 'k', linewidth=2,color="red")
  
  ax[4,3].set_xlabel("10000-64-"+features[incr])
  ax[4,3].set_ylabel('Count')
  ax[4,3].set_title("Fit Values: {:.2f} and {:.2f}".format(mu, std))
  # Create a legend 
  first_legend = ax[4,3].legend(handles=[line2], loc='upper right')


  fig.set_size_inches(18, 12)
  fig.suptitle(features[incr]+' presented in each data_size for each data_rate', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9) 
  incr+= 1


# In[12]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

#dataframe for low_class
avg_stddev_full_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_full_exp_low_df = pd.DataFrame(avg_stddev_full_exp_low)

#dataframe for mid_class
avg_stddev_full_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_full_exp_mid_df = pd.DataFrame(avg_stddev_full_exp_mid)

#dataframe for hig_class
avg_stddev_full_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
avg_stddev_full_exp_hig_df = pd.DataFrame(avg_stddev_full_exp_hig)

#dataframe for FINAL_EXP_LOW
final_exp_low = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
final_exp_low_df = pd.DataFrame(final_exp_low)

#dataframe for FINAL_EXP_MID
final_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
final_exp_mid_df = pd.DataFrame(final_exp_mid)

#dataframe for FINAL_EXP_HIG
final_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
final_exp_hig_df = pd.DataFrame(final_exp_hig)


incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(full_exp_low_class["500-128-"+features[incr]].mean(),full_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-128" , 'Avg': full_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-128', xy =(full_exp_low_class["500-128-"+features[incr]].mean(), full_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-128-"+features[incr]].mean(), full_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-1400-"+features[incr]].mean(),full_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': full_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(full_exp_low_class["500-1400-"+features[incr]].mean(), full_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-1400-"+features[incr]].mean(), full_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': full_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': full_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': full_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-1400-"+features[incr]].mean(),  full_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': full_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(full_exp_low_class["2500-1400-"+features[incr]].mean(), full_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-1400-"+features[incr]].mean(), full_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': full_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': full_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_low_df = avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("LOW")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_full_exp_low_df["Avg"]), 'Stddev': avg_stddev_full_exp_low_df["Stddev"].min()
                ,'CI_lower': avg_stddev_full_exp_low_df["CI_lower"].min(),'CI_upper': avg_stddev_full_exp_low_df["CI_upper"].max()}
  final_exp_low_df = final_exp_low_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[0].axvline(statistics.median(avg_stddev_full_exp_low_df["Avg"]), color ='yellow')

  #middle class
  ax[1].scatter(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': full_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_mid_df = avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': full_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_mid_df = avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-512" , 'Avg': full_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_mid_df = avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': full_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_mid_df = avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_full_exp_mid_df["Avg"]), 'Stddev': avg_stddev_full_exp_mid_df["Stddev"].min()
                ,'CI_lower': avg_stddev_full_exp_mid_df["CI_lower"].min(),'CI_upper': avg_stddev_full_exp_mid_df["CI_upper"].max()}
  final_exp_mid_df = final_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': full_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}  
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': full_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': full_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': full_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': full_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': full_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': full_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': full_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  avg_stddev_full_exp_hig_df = avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_full_exp_hig_df["Avg"]), 'Stddev': avg_stddev_full_exp_hig_df["Stddev"].min()
                ,'CI_lower': avg_stddev_full_exp_hig_df["CI_lower"].min(),'CI_upper': avg_stddev_full_exp_hig_df["CI_upper"].max()}
  final_exp_hig_df = final_exp_hig_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(avg_stddev_full_exp_low_df["Avg"].tail(8), avg_stddev_full_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(avg_stddev_full_exp_mid_df["Avg"].tail(4), avg_stddev_full_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(avg_stddev_full_exp_hig_df["Avg"].tail(8), avg_stddev_full_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')

  #Interpolation

  #dataframe
  tmp = {'NameofExp':[],'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  tmp_df = pd.DataFrame(tmp)

  tmp_df=tmp_df.append([avg_stddev_full_exp_mid_df.tail(8), avg_stddev_full_exp_mid_df.tail(4),avg_stddev_full_exp_hig_df.tail(8)], ignore_index=True,sort=False)
  x=tmp_df["Avg"]
  y=tmp_df["Stddev"]
  #f1 = interp1d(x, y, kind='linear')
  #f1 = interp1d(x.drop_duplicates(), y.drop_duplicates(), kind='quadratic')
  #f1 = interp1d(x.drop_duplicates(), y.drop_duplicates(), kind='cubic')
  #f1=make_interp_spline((x.sort_values()).drop_duplicates(), (y.sort_values()).drop_duplicates())
  

  xnew = np.linspace(tmp_df["Avg"].min(), tmp_df["Avg"].max(), num=1000, endpoint=True)


  #linef1, = ax[3].plot(xnew, f1(xnew))

  # Create a legend 
  #first_legend = ax[3].legend(handles=[linef1], loc='upper right')

  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1

#Saving final_exp_mid_df into csv
final_exp_low_df.to_csv("final_exp_low.csv")
final_exp_mid_df.to_csv("final_exp_mid.csv")
final_exp_hig_df.to_csv("final_exp_hig.csv")


# In[68]:


# Saving of the Pertinant Components characteristics

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]
#dataframe for low_class
PC_avg_stddev_full_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_full_exp_low_df = pd.DataFrame(PC_avg_stddev_full_exp_low)

#dataframe for mid_class
PC_avg_stddev_full_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_full_exp_mid_df = pd.DataFrame(PC_avg_stddev_full_exp_mid)

#dataframe for hig_class
PC_avg_stddev_full_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
PC_avg_stddev_full_exp_hig_df = pd.DataFrame(PC_avg_stddev_full_exp_hig)

#dataframe for FINAL_EXP_LOW
PC_final_exp_low = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
PC_final_exp_low_df = pd.DataFrame(PC_final_exp_low)

#dataframe for FINAL_EXP_MID
PC_final_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[] , 'CI_lower':[], 'CI_upper':[]}
# creating a dataframe from dictionary
PC_final_exp_mid_df = pd.DataFrame(PC_final_exp_mid)

#dataframe for FINAL_EXP_HIG
PC_final_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_final_exp_hig_df = pd.DataFrame(PC_final_exp_hig)


incr =0
for incr in range(0,7):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(full_exp_low_class["500-128-"+features[incr]].mean(),full_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-128" , 'Avg': full_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-128', xy =(full_exp_low_class["500-128-"+features[incr]].mean(), full_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-128-"+features[incr]].mean(), full_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-1400-"+features[incr]].mean(),full_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': full_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(full_exp_low_class["500-1400-"+features[incr]].mean(), full_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-1400-"+features[incr]].mean(), full_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': full_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-512-"+features[incr]].mean(), full_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': full_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': full_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(full_exp_low_class["500-64-"+features[incr]].mean(), full_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': full_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-128-"+features[incr]].mean(), full_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-1400-"+features[incr]].mean(),  full_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': full_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(full_exp_low_class["2500-1400-"+features[incr]].mean(), full_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-1400-"+features[incr]].mean(), full_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': full_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-512-"+features[incr]].mean(), full_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= full_exp_low_class["2500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': full_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': full_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_low_df = PC_avg_stddev_full_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(full_exp_low_class["2500-64-"+features[incr]].mean(), full_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("LOW")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_full_exp_low_df["Avg"]), 'Stddev': PC_avg_stddev_full_exp_low_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_full_exp_low_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_full_exp_low_df["CI_upper"].max()}
  PC_final_exp_low_df = PC_final_exp_low_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[0].axvline(statistics.median(avg_stddev_full_exp_low_df["Avg"]), color ='yellow')

  #middle class
  ax[1].scatter(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': full_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_mid_df = PC_avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-128-"+features[incr]].mean(), full_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': full_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_mid_df = PC_avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-1400-"+features[incr]].mean(), full_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-512" , 'Avg': full_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_mid_df = PC_avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-512-"+features[incr]].mean(), full_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= full_exp_mid_class["5000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': full_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': full_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_mid_df = PC_avg_stddev_full_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(full_exp_mid_class["5000-64-"+features[incr]].mean(), full_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_full_exp_mid_df["Avg"]), 'Stddev': PC_avg_stddev_full_exp_mid_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_full_exp_mid_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_full_exp_mid_df["CI_upper"].max()}
  PC_final_exp_mid_df = PC_final_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': full_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}  
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-128-"+features[incr]].mean(), full_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': full_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-1400-"+features[incr]].mean(), full_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': full_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-512-"+features[incr]].mean(), full_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["7500-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': full_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(full_exp_hig_class["7500-64-"+features[incr]].mean(), full_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': full_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-128-"+features[incr]].mean(), full_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-1400-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': full_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-1400-"+features[incr]].mean(), full_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-512-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': full_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-512-"+features[incr]].mean(), full_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= full_exp_hig_class["10000-64-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': full_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': full_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u}
  PC_avg_stddev_full_exp_hig_df = PC_avg_stddev_full_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(full_exp_hig_class["10000-64-"+features[incr]].mean(), full_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_full_exp_hig_df["Avg"]), 'Stddev': PC_avg_stddev_full_exp_hig_df["Stddev"].min()
                ,'CI_lower': PC_avg_stddev_full_exp_hig_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_full_exp_hig_df["CI_upper"].max()}
  PC_final_exp_hig_df = PC_final_exp_hig_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(PC_avg_stddev_full_exp_low_df["Avg"].tail(8), PC_avg_stddev_full_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(PC_avg_stddev_full_exp_mid_df["Avg"].tail(4), PC_avg_stddev_full_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(PC_avg_stddev_full_exp_hig_df["Avg"].tail(8), PC_avg_stddev_full_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')

  #Interpolation

  #dataframe
  tmp = {'NameofExp':[],'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  tmp_df = pd.DataFrame(tmp)

  tmp_df=tmp_df.append([PC_avg_stddev_full_exp_mid_df.tail(8), PC_avg_stddev_full_exp_mid_df.tail(4),PC_avg_stddev_full_exp_hig_df.tail(8)], ignore_index=True,sort=False)
  x=tmp_df["Avg"]
  y=tmp_df["Stddev"]
  #f1 = interp1d(x, y, kind='linear')
  #f1 = interp1d(x.drop_duplicates(), y.drop_duplicates(), kind='quadratic')
  #f1 = interp1d(x.drop_duplicates(), y.drop_duplicates(), kind='cubic')
  #f1=make_interp_spline((x.sort_values()).drop_duplicates(), (y.sort_values()).drop_duplicates())
  

  xnew = np.linspace(tmp_df["Avg"].min(), tmp_df["Avg"].max(), num=1000, endpoint=True)


  #linef1, = ax[3].plot(xnew, f1(xnew))

  # Create a legend 
  #first_legend = ax[3].legend(handles=[linef1], loc='upper right')

  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1

#Saving final_exp_mid_df into csv
PC_final_exp_low_df.to_csv("PC_final_exp_low.csv")
PC_final_exp_mid_df.to_csv("PC_final_exp_mid.csv")
PC_final_exp_hig_df.to_csv("PC_final_exp_hig.csv")


# In[14]:


between_exp_low_class = low_class.iloc[15:21,:]
between_exp_mid_class = med_class.iloc[15:21,:]
between_exp_hig_class = hig_class.iloc[15:21,:]

#change_rate_{rate}_{packets_size}_bytes

#change_rate_500_128_bytes
change_rate_500_128_bytes =  between_exp_low_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  change_rate_500_128_bytes =  pd.concat([change_rate_500_128_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_500_128_bytes

#change_rate_500_1400_bytes
change_rate_500_1400_bytes = between_exp_low_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  change_rate_500_1400_bytes =  pd.concat([change_rate_500_1400_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_500_1400_bytes


#change_rate_500_512_bytes
change_rate_500_512_bytes = between_exp_low_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  change_rate_500_512_bytes =  pd.concat([change_rate_500_512_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_500_512_bytes

#change_rate_500_64_bytes
change_rate_500_64_bytes = between_exp_low_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  change_rate_500_64_bytes =  pd.concat([change_rate_500_64_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_500_64_bytes


#change_rate_2500_128_bytes
change_rate_2500_128_bytes =  between_exp_low_class.iloc[:,124]
incr=124
for x in range(0,30):
  incr= incr+4
  change_rate_2500_128_bytes =  pd.concat([change_rate_2500_128_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_2500_128_bytes

#change_rate_2500_1400_bytes
change_rate_2500_1400_bytes = between_exp_low_class.iloc[:,125]
incr=125
for x in range(0,30):
  incr= incr+4
  change_rate_2500_1400_bytes =  pd.concat([change_rate_2500_1400_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_2500_1400_bytes


#change_rate_2500_512_bytes
change_rate_2500_512_bytes = between_exp_low_class.iloc[:,126]
incr=126
for x in range(0,30):
  incr= incr+4
  change_rate_2500_512_bytes =  pd.concat([change_rate_2500_512_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_2500_512_bytes

#change_rate_2500_64_bytes
change_rate_2500_64_bytes = between_exp_low_class.iloc[:,127]
incr=127
for x in range(0,30):
  incr= incr+4
  change_rate_2500_64_bytes =  pd.concat([change_rate_2500_64_bytes, between_exp_low_class.iloc[:,incr]],axis=1)

change_rate_2500_64_bytes

#change_rate_5000_128_bytes
change_rate_5000_128_bytes =  between_exp_mid_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  change_rate_5000_128_bytes =  pd.concat([change_rate_5000_128_bytes, between_exp_mid_class.iloc[:,incr]],axis=1)

change_rate_5000_128_bytes

#change_rate_5000_1400_bytes
change_rate_5000_1400_bytes = between_exp_mid_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  change_rate_5000_1400_bytes =  pd.concat([change_rate_5000_1400_bytes, between_exp_mid_class.iloc[:,incr]],axis=1)

change_rate_5000_1400_bytes


#change_rate_5000_512_bytes
change_rate_5000_512_bytes = between_exp_mid_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  change_rate_5000_512_bytes =  pd.concat([change_rate_5000_512_bytes, between_exp_mid_class.iloc[:,incr]],axis=1)

change_rate_5000_512_bytes

#change_rate_5000_64_bytes
change_rate_5000_64_bytes = between_exp_mid_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  change_rate_5000_64_bytes =  pd.concat([change_rate_5000_64_bytes, between_exp_mid_class.iloc[:,incr]],axis=1)

change_rate_5000_64_bytes


#change_rate_7500_128_bytes
change_rate_7500_128_bytes =  between_exp_hig_class.iloc[:,0]
incr=0
for x in range(0,30):
  incr= incr+4
  change_rate_7500_128_bytes =  pd.concat([change_rate_7500_128_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_7500_128_bytes

#change_rate_7500_1400_bytes
change_rate_7500_1400_bytes = between_exp_hig_class.iloc[:,1]
incr=1
for x in range(0,30):
  incr= incr+4
  change_rate_7500_1400_bytes =  pd.concat([change_rate_7500_1400_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_7500_1400_bytes


#change_rate_7500_512_bytes
change_rate_7500_512_bytes = between_exp_hig_class.iloc[:,2]
incr=2
for x in range(0,30):
  incr= incr+4
  change_rate_7500_512_bytes =  pd.concat([change_rate_7500_512_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_7500_512_bytes

#change_rate_7500_64_bytes
change_rate_7500_64_bytes = between_exp_hig_class.iloc[:,3]
incr=3
for x in range(0,30):
  incr= incr+4
  change_rate_7500_64_bytes =  pd.concat([change_rate_7500_64_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_7500_64_bytes

#change_rate_10000_128_bytes
change_rate_10000_128_bytes =  between_exp_hig_class.iloc[:,124]
incr=124
for x in range(0,30):
  incr= incr+4
  change_rate_10000_128_bytes =  pd.concat([change_rate_10000_128_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_10000_128_bytes

#change_rate_10000_1400_bytes
change_rate_10000_1400_bytes = between_exp_hig_class.iloc[:,125]
incr=125
for x in range(0,30):
  incr= incr+4
  change_rate_10000_1400_bytes =  pd.concat([change_rate_10000_1400_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_10000_1400_bytes


#change_rate_10000_512_bytes
change_rate_10000_512_bytes = between_exp_hig_class.iloc[:,126]
incr=126
for x in range(0,30):
  incr= incr+4
  change_rate_10000_512_bytes =  pd.concat([change_rate_10000_512_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_10000_512_bytes

#change_rate_10000_64_bytes
change_rate_10000_64_bytes = between_exp_hig_class.iloc[:,127]
incr=127
for x in range(0,30):
  incr= incr+4
  change_rate_10000_64_bytes =  pd.concat([change_rate_10000_64_bytes, between_exp_hig_class.iloc[:,incr]],axis=1)

change_rate_10000_64_bytes


# In[31]:


#Change part in low class

#For 500 data_rate with differents data_size
fig, ax = plt.subplots(4,31)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  ax[0,incr].plot(change_rate_500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(change_rate_500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(change_rate_500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(change_rate_500_64_bytes.iloc[:,incr],color='orange')
  #ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('500 date_rate at 128,1400,512,64 data_size', fontsize=16)


#For 2500 data_rate with differents data_size
fig, ax = plt.subplots(4,31)
incr =0
for incr in range(0,30):
  ax[0,incr].plot(change_rate_2500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(change_rate_2500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(change_rate_2500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(change_rate_2500_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('2500 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[32]:


#Change part in mid class

#For 5000 data_rate with differents data_size
fig, ax = plt.subplots(4,30)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  ax[0,incr].plot(change_rate_5000_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(change_rate_5000_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(change_rate_5000_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(change_rate_5000_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('5000 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[34]:


#For 7500 data_rate with differents data_size
fig, ax = plt.subplots(4,30)
features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

incr =0
for incr in range(0,30):
  ax[0,incr].plot(change_rate_7500_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(change_rate_7500_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(change_rate_7500_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(change_rate_7500_64_bytes.iloc[:,incr],color='orange')
  ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('7500 date_rate at 128,1400,512,64 data_size', fontsize=16)


#For 10000 data_rate with differents data_size
fig, ax = plt.subplots(4,30)
incr =0
for incr in range(0,30):
  ax[0,incr].plot(change_rate_10000_128_bytes.iloc[:,incr])
  ax[0,incr].set_ylabel(features[incr])
  incr=incr+1
  
incr =0
for incr in range(0,30):
  ax[1,incr].plot(change_rate_10000_1400_bytes.iloc[:,incr],color='green')
  ax[1,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[2,incr].plot(change_rate_10000_512_bytes.iloc[:,incr],color='red')
  ax[2,incr].set_ylabel(features[incr])
  incr=incr+1

incr =0
for incr in range(0,30):
  ax[3,incr].plot(change_rate_10000_64_bytes.iloc[:,incr],color='orange')
  #ax[3,incr].set_xlabel('Seconds')
  ax[3,incr].set_ylabel(features[incr])
  incr=incr+1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(50, 18)
fig.suptitle('10000 date_rate at 128,1400,512,64 data_size', fontsize=16)


# In[15]:


between_exp_low_class = low_class.iloc[15:21,:]
between_exp_mid_class = med_class.iloc[15:21,:]
between_exp_hig_class = hig_class.iloc[15:21,:]


# In[16]:


#General dataframe containing average and stddev of full between_exp

avg_stddev_between_exp = {'NameofExp':[], 'Avg':[],'Stddev':[]}
# creating a dataframe from dictionary
avg_stddev_between_exp_df = pd.DataFrame(avg_stddev_between_exp)

features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]
incr =0
for incr in range(0,30):
  #low class
  #dataframe for low_class
  avg_stddev_between_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  avg_stddev_between_exp_low_df = pd.DataFrame(avg_stddev_between_exp_low)

  new_element = {'NameofExp':"500-128" , 'Avg': between_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-128-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"500-1400" , 'Avg': between_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-1400-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"500-512" , 'Avg': between_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-512-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"500-64" , 'Avg': between_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-64-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"2500-128" , 'Avg': between_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-128-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"2500-1400" , 'Avg': between_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-1400-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"2500-512" , 'Avg': between_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-512-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"2500-64" , 'Avg': between_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-64-"+features[incr]].std()}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  
  # avg and stddev saving for global exp 
  avg_stddev_between_exp_df.append(avg_stddev_between_exp_low_df, ignore_index=True)


  #middle class
  #for mid_class
  avg_stddev_between_exp_mid = {'NameofExp':[], 'Avg':[],'Stddev':[] }
  # creating a dataframe from dictionary
  avg_stddev_between_exp_mid_df = pd.DataFrame(avg_stddev_between_exp_mid)

  new_element = {'NameofExp':"5000-128" , 'Avg': between_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-128-"+features[incr]].std()}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"5000-1400" , 'Avg': between_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-1400-"+features[incr]].std()}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"5000-512" , 'Avg': between_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-512-"+features[incr]].std()}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"5000-64" , 'Avg': between_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-64-"+features[incr]].std()}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  
  # avg and stddev saving for global exp 
  avg_stddev_between_exp_df.append(avg_stddev_between_exp_mid_df, ignore_index=True)

  #hig class
  #for hig_class
  avg_stddev_between_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[]}
  # creating a dataframe from dictionary
  avg_stddev_between_exp_hig_df = pd.DataFrame(avg_stddev_between_exp_hig)

  new_element = {'NameofExp':"7500-128" , 'Avg': between_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-128-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"7500-1400" , 'Avg': between_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-1400-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"7500-512" , 'Avg': between_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-512-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"7500-64" , 'Avg': between_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-64-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"10000-128" , 'Avg': between_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-128-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"10000-1400" , 'Avg': between_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-1400-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"10000-512" , 'Avg': between_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-512-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  new_element = {'NameofExp':"10000-64" , 'Avg': between_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-64-"+features[incr]].std()}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  
  # avg and stddev saving for global exp 
  avg_stddev_between_exp_df.append(avg_stddev_between_exp_hig_df, ignore_index=True)

  incr+= 1

#Plot
fig, ax = plt.subplots(1,3)
ax[0].scatter(avg_stddev_between_exp_low_df['Avg'], avg_stddev_between_exp_low_df['Stddev'], facecolor ='blue')
ax[0].set_xlabel('avg')
ax[0].set_ylabel('stddev')
ax[0].set_ylim(0,)
ax[1].scatter(avg_stddev_between_exp_mid_df['Avg'], avg_stddev_between_exp_mid_df['Stddev'], facecolor ='red')
ax[1].set_xlabel('avg')
ax[1].set_ylabel('stddev')
ax[1].set_ylim(0,)
ax[2].scatter(avg_stddev_between_exp_hig_df['Avg'], avg_stddev_between_exp_hig_df['Stddev'], facecolor ='green')
ax[2].set_xlabel('avg')
ax[2].set_ylabel('stddev')
ax[2].set_ylim(0,)

ax[0].set_title("stddev(avg) in low_class")
ax[1].set_title("stddev(avg) in mid_class")
ax[2].set_title("stddev(avg) in hig_class")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(16, 10)
fig.suptitle('stddev(avg) of the full shape in Between_EXP', fontsize=16)
plt.show()


# In[17]:


features =["branches","branch-load-misses","branch-misses","bus-cycles","cache-misses","cache-references","context-switches","cpu-clock","cycles","dTLB-load-misses","dTLB-store-misses","dTLB-stores","instructions","iTLB-load-misses","iTLB-loads","L1-dcache-load-misses","L1-dcache-loads","L1-dcache-stores","L1-icache-load-misses","LLC-load-misses","LLC-loads","LLC-store-misses","LLC-stores","minor-faults","node-load-misses","node-loads","node-store-misses","node-stores","page-faults","ref-cycles","task-clock"]

#dataframe for low_class
avg_stddev_between_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_between_exp_low_df = pd.DataFrame(avg_stddev_between_exp_low)

#dataframe for mid_class
avg_stddev_between_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[]  }
# creating a dataframe from dictionary
avg_stddev_full_between_mid_df = pd.DataFrame(avg_stddev_between_mid)

#dataframe for hig_class
avg_stddev_between_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
avg_stddev_between_exp_hig_df = pd.DataFrame(avg_stddev_between_exp_hig)

#dataframe for BETWEEN_EXP_LOW
between_exp_low = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
between_exp_low_df = pd.DataFrame(between_exp_low)

#dataframe for BETWEEN_EXP_MID
between_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[]  }
# creating a dataframe from dictionary
between_exp_mid_df = pd.DataFrame(between_exp_mid)

#dataframe for BETWEEN_EXP_HIG
between_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[],'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
between_exp_hig_df = pd.DataFrame(between_exp_hig)


incr =0
for incr in range(0,30):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(between_exp_low_class["500-128-"+features[incr]].mean(),between_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
  new_element = {'NameofExp':"500-128" , 'Avg': between_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u }
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)

  ax[0].annotate('500-128', xy =(between_exp_low_class["500-128-"+features[incr]].mean(), between_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-128-"+features[incr]].mean(), between_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-1400-"+features[incr]].mean(),between_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': between_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(between_exp_low_class["500-1400-"+features[incr]].mean(), between_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-1400-"+features[incr]].mean(), between_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': between_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': between_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': between_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-1400-"+features[incr]].mean(),  between_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': between_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(between_exp_low_class["2500-1400-"+features[incr]].mean(), between_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-1400-"+features[incr]].mean(), between_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': between_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': between_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_low_df = avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("LOW")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_between_exp_low_df["Avg"]), 'Stddev': avg_stddev_between_exp_low_df["Stddev"].max()
                ,'CI_lower':avg_stddev_between_exp_low_df["CI_lower"].min(),'CI_upper':avg_stddev_between_exp_low_df["CI_upper"].max()}
  between_exp_low_df = between_exp_low_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[0].axvline(statistics.median(avg_stddev_full_exp_low_df["Avg"]), color ='yellow')

  #middle class
  ax[1].scatter(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': between_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': between_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-512" , 'Avg': between_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': between_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_mid_df = avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_between_exp_mid_df["Avg"]), 'Stddev': avg_stddev_between_exp_mid_df["Stddev"].max()
                ,'CI_lower':avg_stddev_between_exp_mid_df["CI_lower"].min(),'CI_upper':avg_stddev_between_exp_mid_df["CI_upper"].max()}
  between_exp_mid_df = between_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': between_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': between_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': between_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': between_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': between_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': between_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': between_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': between_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  avg_stddev_between_exp_hig_df = avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(avg_stddev_between_exp_hig_df["Avg"]), 'Stddev': avg_stddev_between_exp_hig_df["Stddev"].max()
                ,'CI_lower':avg_stddev_between_exp_hig_df["CI_lower"].min(),'CI_upper':avg_stddev_between_exp_hig_df["CI_upper"].max()}
  between_exp_hig_df = between_exp_hig_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(avg_stddev_between_exp_low_df["Avg"].tail(8), avg_stddev_between_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(avg_stddev_between_exp_mid_df["Avg"].tail(4), avg_stddev_between_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(avg_stddev_between_exp_hig_df["Avg"].tail(8), avg_stddev_between_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')
  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1


#Saving final_exp_mid_df into csv
between_exp_low_df.to_csv("between_exp_low.csv")
between_exp_mid_df.to_csv("between_exp_mid.csv")
between_exp_hig_df.to_csv("between_exp_hig.csv")


# In[18]:


# Saving of Pertinant features characteristics : between_exp

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

#dataframe for low_class
PC_avg_stddev_between_exp_low = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_between_exp_low_df = pd.DataFrame(PC_avg_stddev_between_exp_low)

#dataframe for mid_class
PC_avg_stddev_between_mid = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[]  }
# creating a dataframe from dictionary
PC_avg_stddev_between_exp_mid_df = pd.DataFrame(PC_avg_stddev_between_mid)

#dataframe for hig_class
PC_avg_stddev_between_exp_hig = {'NameofExp':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_avg_stddev_between_exp_hig_df = pd.DataFrame(PC_avg_stddev_between_exp_hig)

#dataframe for BETWEEN_EXP_LOW
PC_between_exp_low = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_between_exp_low_df = pd.DataFrame(PC_between_exp_low)

#dataframe for BETWEEN_EXP_MID
PC_between_exp_mid = {'Feature':[], 'Avg':[],'Stddev':[], 'CI_lower':[], 'CI_upper':[]  }
# creating a dataframe from dictionary
PC_between_exp_mid_df = pd.DataFrame(PC_between_exp_mid)

#dataframe for BETWEEN_EXP_HIG
PC_between_exp_hig = {'Feature':[], 'Avg':[],'Stddev':[],'CI_lower':[], 'CI_upper':[] }
# creating a dataframe from dictionary
PC_between_exp_hig_df = pd.DataFrame(PC_between_exp_hig)


incr =0
for incr in range(0,7):
  fig, ax = plt.subplots(1,4)
  #low class
  ax[0].scatter(between_exp_low_class["500-128-"+features[incr]].mean(),between_exp_low_class["500-128-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-128-"+features[incr]]
  CI_l , CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
  new_element = {'NameofExp':"500-128" , 'Avg': between_exp_low_class["500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-128-"+features[incr]].std()
                ,'CI_lower': CI_l,'CI_upper': CI_u }
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)

  ax[0].annotate('500-128', xy =(between_exp_low_class["500-128-"+features[incr]].mean(), between_exp_low_class["500-128-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-128-"+features[incr]].mean(), between_exp_low_class["500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-1400-"+features[incr]].mean(),between_exp_low_class["500-1400-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-1400" , 'Avg': between_exp_low_class["500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-1400', xy =(between_exp_low_class["500-1400-"+features[incr]].mean(), between_exp_low_class["500-1400-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-1400-"+features[incr]].mean(), between_exp_low_class["500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-512" , 'Avg': between_exp_low_class["500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-512', xy =(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-512-"+features[incr]].mean(), between_exp_low_class["500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std(), facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"500-64" , 'Avg': between_exp_low_class["500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('500-64', xy =(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std()),
             xytext =(between_exp_low_class["500-64-"+features[incr]].mean(), between_exp_low_class["500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-128" , 'Avg': between_exp_low_class["2500-128-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-128', xy =(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-128-"+features[incr]].mean(), between_exp_low_class["2500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-1400-"+features[incr]].mean(),  between_exp_low_class["2500-1400-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-1400" , 'Avg': between_exp_low_class["2500-1400-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-1400', xy =(between_exp_low_class["2500-1400-"+features[incr]].mean(), between_exp_low_class["2500-1400-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-1400-"+features[incr]].mean(), between_exp_low_class["2500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-512" , 'Avg': between_exp_low_class["2500-512-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-512', xy =(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-512-"+features[incr]].mean(), between_exp_low_class["2500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')

  ax[0].scatter(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std(),facecolor ='blue')
  #create 95% confidence interval for population mean weight
  data= between_exp_low_class["2500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"2500-64" , 'Avg': between_exp_low_class["2500-64-"+features[incr]].mean(), 'Stddev': between_exp_low_class["2500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_low_df = PC_avg_stddev_between_exp_low_df.append(new_element, ignore_index = True)
  ax[0].annotate('2500-64', xy =(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std()),
             xytext =(between_exp_low_class["2500-64-"+features[incr]].mean(), between_exp_low_class["2500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='blue',
                               shrink = 0.05),   )
  ax[0].set_xlabel("Avg")
  ax[0].set_ylabel('Stddev')
  ax[0].set_title("LOW")

  #Saving std dev per class : here low
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_between_exp_low_df["Avg"]), 'Stddev': PC_avg_stddev_between_exp_low_df["Stddev"].max()
                ,'CI_lower':PC_avg_stddev_between_exp_low_df["CI_lower"].min(),'CI_upper':PC_avg_stddev_between_exp_low_df["CI_upper"].max()}
  PC_between_exp_low_df = PC_between_exp_low_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[0].axvline(statistics.median(avg_stddev_full_exp_low_df["Avg"]), color ='yellow')

  #middle class
  ax[1].scatter(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-128" , 'Avg': between_exp_mid_class["5000-128-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_mid_df = PC_avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-128', xy =(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-128-"+features[incr]].mean(), between_exp_mid_class["5000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-1400" , 'Avg': between_exp_mid_class["5000-1400-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_mid_df = PC_avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-1400', xy =(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-1400-"+features[incr]].mean(), between_exp_mid_class["5000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  
  ax[1].scatter(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-512" , 'Avg': between_exp_mid_class["5000-512-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_mid_df = PC_avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-512', xy =(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-512-"+features[incr]].mean(), between_exp_mid_class["5000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),  )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')

  ax[1].scatter(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std(),facecolor ='red')
  #create 95% confidence interval for population mean weight
  data= between_exp_mid_class["5000-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"5000-64" , 'Avg': between_exp_mid_class["5000-64-"+features[incr]].mean(), 'Stddev': between_exp_mid_class["5000-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_mid_df = PC_avg_stddev_between_exp_mid_df.append(new_element, ignore_index = True)
  ax[1].annotate('5000-64', xy =(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std()),
             xytext =(between_exp_mid_class["5000-64-"+features[incr]].mean(), between_exp_mid_class["5000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='red',
                               shrink = 0.05),   )
  ax[1].set_xlabel("Avg")
  ax[1].set_ylabel('Stddev')
  ax[1].set_title("MID")

  #Saving std dev per class : here mid
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_between_exp_mid_df["Avg"]), 'Stddev': PC_avg_stddev_between_exp_mid_df["Stddev"].max()
                ,'CI_lower': PC_avg_stddev_between_exp_mid_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_between_exp_mid_df["CI_upper"].max()}
  PC_between_exp_mid_df = PC_between_exp_mid_df.append(new_element, ignore_index = True)

  #Plot of the mean
  #ax[1].axvline(statistics.median(avg_stddev_full_exp_mid_df["Avg"]), color ='yellow')

  #hig class

  ax[2].scatter(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-128" , 'Avg': between_exp_hig_class["7500-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-128', xy =(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-128-"+features[incr]].mean(), between_exp_hig_class["7500-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-1400" , 'Avg': between_exp_hig_class["7500-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-1400', xy =(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-1400-"+features[incr]].mean(), between_exp_hig_class["7500-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-512" , 'Avg': between_exp_hig_class["7500-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-512', xy =(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-512-"+features[incr]].mean(), between_exp_hig_class["7500-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["7500-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"7500-64" , 'Avg': between_exp_hig_class["7500-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["7500-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('7500-64', xy =(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std()),
             xytext =(between_exp_hig_class["7500-64-"+features[incr]].mean(), between_exp_hig_class["7500-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  

  ax[2].scatter(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-128-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-128" , 'Avg': between_exp_hig_class["10000-128-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-128-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-128', xy =(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-128-"+features[incr]].mean(), between_exp_hig_class["10000-128-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  
  ax[2].scatter(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-1400-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-1400" , 'Avg': between_exp_hig_class["10000-1400-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-1400-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-1400', xy =(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-1400-"+features[incr]].mean(), between_exp_hig_class["10000-1400-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-512-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-512" , 'Avg': between_exp_hig_class["10000-512-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-512-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-512', xy =(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-512-"+features[incr]].mean(), between_exp_hig_class["10000-512-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')

  ax[2].scatter(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std(),facecolor ='green')
  #create 95% confidence interval for population mean weight
  data= between_exp_hig_class["10000-64-"+features[incr]]
  CI_l,CI_u=st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  new_element = {'NameofExp':"10000-64" , 'Avg': between_exp_hig_class["10000-64-"+features[incr]].mean(), 'Stddev': between_exp_hig_class["10000-64-"+features[incr]].std()
                ,'CI_lower':CI_l,'CI_upper':CI_u}
  PC_avg_stddev_between_exp_hig_df = PC_avg_stddev_between_exp_hig_df.append(new_element, ignore_index = True)
  ax[2].annotate('10000-64', xy =(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std()),
             xytext =(between_exp_hig_class["10000-64-"+features[incr]].mean(), between_exp_hig_class["10000-64-"+features[incr]].std()),
             arrowprops = dict(facecolor ='green',
                               shrink = 0.05),   )
  ax[2].set_xlabel("Avg")
  ax[2].set_ylabel('Stddev')
  ax[2].set_title("HIGH")

  #Saving std dev per class : here hig
  new_element = {'Feature':features[incr] , 'Avg': statistics.median(PC_avg_stddev_between_exp_hig_df["Avg"]), 'Stddev': PC_avg_stddev_between_exp_hig_df["Stddev"].max()
                ,'CI_lower': PC_avg_stddev_between_exp_hig_df["CI_lower"].min(),'CI_upper': PC_avg_stddev_between_exp_hig_df["CI_upper"].max()}
  PC_between_exp_hig_df = PC_between_exp_hig_df.append(new_element, ignore_index = True)


  #Plot of the mean
  #ax[2].axvline(statistics.median(avg_stddev_full_exp_hig_df["Avg"]), color ='yellow')

  #Scatter avg and std groupby class
  ax[3].scatter(PC_avg_stddev_between_exp_low_df["Avg"].tail(8), PC_avg_stddev_between_exp_low_df["Stddev"].tail(8),label='low',facecolor ='blue')
  ax[3].scatter(PC_avg_stddev_between_exp_mid_df["Avg"].tail(4), PC_avg_stddev_between_exp_mid_df["Stddev"].tail(4),label='mid',facecolor ='red')
  ax[3].scatter(PC_avg_stddev_between_exp_hig_df["Avg"].tail(8), PC_avg_stddev_between_exp_hig_df["Stddev"].tail(8),label='hig',facecolor ='green')
  ax[3].set_xlabel("Avg")
  ax[3].set_ylabel('Stddev')
  ax[3].set_title("All classes")
  

  fig.set_size_inches(18, 8)
  fig.suptitle(features[incr]+' presented in LOW-MID-HIG classes', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
  incr+= 1


#Saving final_exp_mid_df into csv
PC_between_exp_low_df.to_csv("PC_between_exp_low.csv")
PC_between_exp_mid_df.to_csv("PC_between_exp_mid.csv")
PC_between_exp_hig_df.to_csv("PC_between_exp_hig.csv")


# In[19]:


# Checking of corrolation beteween classes of ZERO_exp ==> CHANGE_exp
#
#  Corr! low ! mid ! hig !
#  '''''' ''''''''''''''''''     
#  low !  ?  !  ?  !  ?  !
#  mid !  ?  !  ?  !  ?  ! 
#  hig !  ?  !  ?  !  ?  !
#      ''''''''''''''''''

def plot_feature_name_on_axe1(ax,exp_name):
    i=0
    for i in range(0,30):
        ax.annotate(exp_name['Feature'].iloc[i], xy =(exp_name['Avg'].iloc[i], exp_name['Stddev'].iloc[i]),
             xytext =(exp_name['Avg'].iloc[i],exp_name['Stddev'].iloc[i]),
             arrowprops = dict(facecolor ='blue',
                                shrink = 0.01),   )
        i=i+1
        
def plot_feature_name_on_axe2(ax,exp_name):
    i=0
    for i in range(0,30):
        ax.annotate(exp_name['Feature'].iloc[i], xy =(exp_name['Avg'].iloc[i], exp_name['Stddev'].iloc[i]),
             xytext =(exp_name['Avg'].iloc[i],exp_name['Stddev'].iloc[i]),
             arrowprops = dict(facecolor ='orange',
                                shrink = 0.01),   )
        i=i+1

avg_stddev_between_exp_mid_df


# In[20]:


# For PC

def PC_plot_feature_name_on_axe1(ax,exp_name):
    i=0
    for i in range(0,7):
        ax.annotate(exp_name['Feature'].iloc[i], xy =(exp_name['Avg'].iloc[i], exp_name['Stddev'].iloc[i]),
             xytext =(exp_name['Avg'].iloc[i],exp_name['Stddev'].iloc[i]),
             arrowprops = dict(facecolor ='blue',
                                shrink = 0.01),   )
        i=i+1
        
def PC_plot_feature_name_on_axe2(ax,exp_name):
    i=0
    for i in range(0,7):
        ax.annotate(exp_name['Feature'].iloc[i], xy =(exp_name['Avg'].iloc[i], exp_name['Stddev'].iloc[i]),
             xytext =(exp_name['Avg'].iloc[i],exp_name['Stddev'].iloc[i]),
             arrowprops = dict(facecolor ='orange',
                                shrink = 0.01),   )
        i=i+1


# In[21]:


between_exp_hig_df
#between_exp_hig_df.drop(between_exp_hig_df.index[-1], inplace=True)
#between_exp_hig_df


# In[43]:


#dataframe for hig_class
zero_to_between = {'Movement':[], 'Pearsonr_Corr':[],'Spearman_Corr':[] }
# creating a dataframe from dictionary
zero_to_between_df = pd.DataFrame(zero_to_between)

#Plot
fig, ax = plt.subplots(3,3)
ax[0,0].scatter(zero_exp_low_df['Avg'], zero_exp_low_df['Stddev'])
ax[0,0].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])

ax[0,0].set_xlabel('avg')
ax[0,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,0].set_title("Zero_LOW to Betw_LOW")
x=zero_exp_low_df["Avg"]
y=between_exp_low_df["Avg"]
#Interpolation plot
line1, =ax[0,0].plot(x,zero_exp_low_df['Stddev'], label="Zero_LOW")
line2, =ax[0,0].plot(y,between_exp_low_df['Stddev'],label="Betw_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,0],zero_exp_low_df)
plot_feature_name_on_axe2(ax[0,0],between_exp_low_df)

ax[0,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Low to Betw_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)


ax[0,1].scatter(zero_exp_low_df['Avg'], zero_exp_low_df['Stddev'])
ax[0,1].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[0,1].set_xlabel('avg')
ax[0,1].set_ylabel('stddev')
#ax[0,1].set_ylim(0,)
ax[0,1].set_title("Zero_LOW to Betw_MID")
x=zero_exp_low_df["Avg"]
y=between_exp_mid_df["Avg"]
#Interpolation plot
line1, = ax[0,1].plot(x,zero_exp_low_df['Stddev'],label="Zero_LOW")
line2, = ax[0,1].plot(y,between_exp_mid_df['Stddev'],label="Betw_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,1],zero_exp_low_df)
plot_feature_name_on_axe2(ax[0,1],between_exp_mid_df)

ax[0,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Low to Betw_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[0,2].scatter(zero_exp_low_df['Avg'], zero_exp_low_df['Stddev'])
ax[0,2].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[0,2].set_xlabel('avg')
ax[0,2].set_ylabel('stddev')
#ax[0,2].set_ylim(0,)
ax[0,2].set_title("Zero_LOW to Betw_HIG")
x=zero_exp_low_df["Avg"]
y=between_exp_hig_df["Avg"]
#Interpolation plot
line1, = ax[0,2].plot(x,zero_exp_low_df['Stddev'],label="Zero_LOW")
line2, = ax[0,2].plot(y,between_exp_hig_df['Stddev'],label="Betw_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,2],zero_exp_low_df)
plot_feature_name_on_axe2(ax[0,2],between_exp_hig_df)

ax[0,2].legend(handles=[line1,line2], loc='upper right')


corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Low to Betw_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)
  
ax[1,0].scatter(zero_exp_mid_df['Avg'], zero_exp_mid_df['Stddev'])
ax[1,0].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])
ax[1,0].set_xlabel('avg')
ax[1,0].set_ylabel('stddev')
#ax[1,0].set_ylim(0,)
ax[1,0].set_title("Zero_MID to Betw_LOW")
x=zero_exp_mid_df["Avg"]
y=between_exp_low_df["Avg"]
#Interpolation plot
line1, = ax[1,0].plot(x,zero_exp_mid_df['Stddev'],label="Zero_LOW")
line2, = ax[1,0].plot(y,between_exp_low_df['Stddev'],label="Betw_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,0],zero_exp_mid_df)
plot_feature_name_on_axe2(ax[1,0],between_exp_low_df)

ax[1,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Mid to Betw_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[1,1].scatter(zero_exp_mid_df['Avg'], zero_exp_mid_df['Stddev'])
ax[1,1].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[1,1].set_xlabel('avg')
ax[1,1].set_ylabel('stddev')
#ax[1,1].set_ylim(0,)
ax[1,1].set_title("Zero_MID to Betw_MID")
x=zero_exp_mid_df["Avg"]
y=between_exp_mid_df["Avg"]

#Interpolation plot
line1, = ax[1,1].plot(x,zero_exp_mid_df['Stddev'],label="Zero_MID")
line2, = ax[1,1].plot(y,between_exp_mid_df['Stddev'],label="Betw_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,1],zero_exp_mid_df)
plot_feature_name_on_axe2(ax[1,1],between_exp_mid_df)

ax[1,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Mid to Betw_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[1,2].scatter(zero_exp_mid_df['Avg'], zero_exp_mid_df['Stddev'])
ax[1,2].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[1,2].set_xlabel('avg')
ax[1,2].set_ylabel('stddev')
#ax[0,2].set_ylim(0,)
ax[1,2].set_title("Zero_MID to Betw_HIG")
x=zero_exp_mid_df["Avg"]
y=between_exp_hig_df["Avg"]
#Interpolation plot
line1, = ax[1,2].plot(x,zero_exp_mid_df['Stddev'],label="Zero_MID")
line2, = ax[1,2].plot(y,between_exp_hig_df['Stddev'],label="Betw_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,2],zero_exp_mid_df)
plot_feature_name_on_axe2(ax[1,2],between_exp_hig_df)

ax[1,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Mid to Betw_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[2,0].scatter(zero_exp_hig_df['Avg'], zero_exp_hig_df['Stddev'])
ax[2,0].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])
ax[2,0].set_xlabel('avg')
ax[2,0].set_ylabel('stddev')
#ax[1,0].set_ylim(0,)
ax[2,0].set_title("Zero_HIG to Betw_LOW")
x=zero_exp_hig_df["Avg"]
y=between_exp_low_df["Avg"]

#Interpolation plot
line1, = ax[2,0].plot(x,zero_exp_hig_df['Stddev'],label="Zero_HIG")
line2, = ax[2,0].plot(y,between_exp_low_df['Stddev'],label="Betw_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,0],zero_exp_hig_df)
plot_feature_name_on_axe2(ax[2,0],between_exp_low_df)

ax[2,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Hig to Betw_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[2,1].scatter(zero_exp_hig_df['Avg'], zero_exp_hig_df['Stddev'])
ax[2,1].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[2,1].set_xlabel('avg')
ax[2,1].set_ylabel('stddev')
#ax[1,1].set_ylim(0,)
ax[2,1].set_title("Zero_HIG to Betw_MID")
x=zero_exp_hig_df["Avg"]
y=between_exp_mid_df["Avg"]

#Interpolation plot
line1, = ax[2,1].plot(x,zero_exp_hig_df['Stddev'],label="Zero_HIG")
line2, = ax[2,1].plot(y,between_exp_mid_df['Stddev'],label="Betw_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,1],zero_exp_hig_df)
plot_feature_name_on_axe2(ax[2,1],between_exp_mid_df)

ax[2,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Hig to Betw_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)

ax[2,2].scatter(zero_exp_hig_df['Avg'], zero_exp_hig_df['Stddev'])
ax[2,2].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[2,2].set_xlabel('avg')
ax[2,2].set_ylabel('stddev')
#ax[0,2].set_ylim(0,)
ax[2,2].set_title("Zero_HIG to Betw_HIG")
x=zero_exp_hig_df["Avg"]
y=between_exp_hig_df["Avg"]

#Interpolation plot
line1, = ax[2,2].plot(x,zero_exp_hig_df['Stddev'],label="Zero_HIG")
line2, = ax[2,2].plot(y,between_exp_hig_df['Stddev'],label="Betw_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,2],zero_exp_hig_df)
plot_feature_name_on_axe2(ax[2,2],between_exp_hig_df)

ax[2,2].legend(handles=[line1,line2], loc='upper right')


corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero_Hig to Betw_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
zero_to_between_df = zero_to_between_df.append(new_element, ignore_index = True)


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(25, 10)
plt.show()

#Saving Correlation into csv
zero_to_between_df.to_csv("zero_to_between.csv")


# In[22]:


# Visualisation of Pertinent features Zero to Between

#dataframe for hig_class
PC_zero_to_between = {'Movement':[], 'Pearsonr_Corr':[],'Spearman_Corr':[] }
# creating a dataframe from dictionary
PC_zero_to_between_df = pd.DataFrame(PC_zero_to_between)

#Plot
fig, ax = plt.subplots(1,3)
ax[0].scatter(PC_zero_exp_df['Avg'], PC_zero_exp_df['Stddev'])
ax[0].scatter(PC_between_exp_low_df['Avg'], PC_between_exp_low_df['Stddev'])

ax[0].set_xlabel('avg')
ax[0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0].set_title("Zero to Betw_LOW")
x=PC_zero_exp_df["Avg"]
y=PC_between_exp_low_df["Avg"]
#Interpolation plot
line1, =ax[0].plot(x,PC_zero_exp_df['Stddev'], label="Zero")
line2, =ax[0].plot(y,PC_between_exp_low_df['Stddev'],label="Betw_LOW")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[0],PC_zero_exp_df)
PC_plot_feature_name_on_axe2(ax[0],PC_between_exp_low_df)

ax[0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero to Betw_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_zero_to_between_df = PC_zero_to_between_df.append(new_element, ignore_index = True)


ax[1].scatter(PC_zero_exp_df['Avg'], PC_zero_exp_df['Stddev'])
ax[1].scatter(PC_between_exp_mid_df['Avg'], PC_between_exp_mid_df['Stddev'])
ax[1].set_xlabel('avg')
ax[1].set_ylabel('stddev')
#ax[0,1].set_ylim(0,)
ax[1].set_title("Zero to Betw_MID")
x=PC_zero_exp_df["Avg"]
y=PC_between_exp_mid_df["Avg"]
#Interpolation plot
line1, = ax[1].plot(x,PC_zero_exp_df['Stddev'],label="Zero")
line2, = ax[1].plot(y,PC_between_exp_mid_df['Stddev'],label="Betw_MID")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[1],PC_zero_exp_df)
PC_plot_feature_name_on_axe2(ax[1],PC_between_exp_mid_df)

ax[1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero to Betw_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_zero_to_between_df = PC_zero_to_between_df.append(new_element, ignore_index = True)

ax[2].scatter(PC_zero_exp_df['Avg'], PC_zero_exp_df['Stddev'])
ax[2].scatter(PC_between_exp_hig_df['Avg'], PC_between_exp_hig_df['Stddev'])
ax[2].set_xlabel('avg')
ax[2].set_ylabel('stddev')
#ax[0,2].set_ylim(0,)
ax[2].set_title("Zero to Betw_HIG")
x=PC_zero_exp_df["Avg"]
y=PC_between_exp_hig_df["Avg"]
#Interpolation plot
line1, = ax[2].plot(x,PC_zero_exp_df['Stddev'],label="Zero")
line2, = ax[2].plot(y,PC_between_exp_hig_df['Stddev'],label="Betw_HIG")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[2],PC_zero_exp_df)
PC_plot_feature_name_on_axe2(ax[2],PC_between_exp_hig_df)

ax[2].legend(handles=[line1,line2], loc='upper right')


corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Zero to Betw_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_zero_to_between_df = PC_zero_to_between_df.append(new_element, ignore_index = True)  

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(25, 10)
plt.show()

#Saving Correlation into csv
PC_zero_to_between_df.to_csv("PC_zero_to_between.csv")


# In[23]:


# Difference Visualisation of Pertinent features Zero to Between

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

#Plot
fig, ax = plt.subplots(1,3)

x= PC_zero_exp_df['Avg']
y= PC_between_exp_low_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[0].bar(xlim -width/2, x, width, label='PC_zero_exp_df')
rects2 = ax[0].bar(xlim +width/2, y, width, label='PC_between_exp_low_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Avg')
ax[0].set_title('Pertinent Features numerical differences')
#ax[2,2].set_xticks(xlim, labels)
ax[0].set_xticklabels(PC_zero_exp_df["Feature"].astype(str).values, rotation='vertical')
ax[0].legend()


x= PC_zero_exp_df['Avg']
y= PC_between_exp_mid_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[1].bar(xlim -width/2, x, width, label='PC_zero_exp_df')
rects2 = ax[1].bar(xlim +width/2, y, width, label='PC_between_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Avg')
ax[1].set_title('Pertinent Features numerical differences')
#ax[2,2].set_xticks(xlim, labels)
ax[1].set_xticklabels(PC_zero_exp_df["Feature"].astype(str).values, rotation='vertical')
ax[1].legend()


x= PC_zero_exp_df['Avg']
y= PC_between_exp_hig_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[2].bar(xlim -width/2, x, width, label='PC_zero_exp_df')
rects2 = ax[2].bar(xlim +width/2, y, width, label='PC_between_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2].set_ylabel('Avg')
ax[2].set_title('Pertinent Features numerical differences')
#ax[2,2].set_xticks(xlim, labels)
ax[2].set_xticklabels(PC_zero_exp_df["Feature"].astype(str).values, rotation='vertical')
ax[2].legend()


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(28, 13)
plt.show()


# In[29]:


# BEtween to Fianal exp
between_to_final = {'Movement':[], 'Pearsonr_Corr':[],'Spearman_Corr':[] }
# creating a dataframe from dictionary
between_to_final_df = pd.DataFrame(between_to_final)

#Plot
fig, ax = plt.subplots(3,3)
ax[0,0].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])
ax[0,0].scatter(final_exp_low_df['Avg'], final_exp_low_df['Stddev'])
ax[0,0].set_xlabel('avg')
ax[0,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,0].set_title("Betw_LOW to Final_LOW")
x=between_exp_low_df["Avg"]
y=final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[0,0].plot(x,between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,0].plot(y,final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,0],between_exp_low_df)
plot_feature_name_on_axe2(ax[0,0],final_exp_low_df)

ax[0,0].legend(handles=[line1,line2], loc='upper right')
corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[0,1].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])
ax[0,1].scatter(final_exp_mid_df['Avg'], final_exp_mid_df['Stddev'])
ax[0,1].set_xlabel('avg')
ax[0,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,1].set_title("Betw_LOW to Final_MID")
x=between_exp_low_df["Avg"]
y=final_exp_mid_df["Avg"]

#Interpolation plot
line1, =ax[0,1].plot(x,between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,1].plot(y,final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,1],between_exp_low_df)
plot_feature_name_on_axe2(ax[0,1],final_exp_mid_df)

ax[0,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[0,2].scatter(between_exp_low_df['Avg'], between_exp_low_df['Stddev'])
ax[0,2].scatter(final_exp_hig_df['Avg'], final_exp_hig_df['Stddev'])
ax[0,2].set_xlabel('avg')
ax[0,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,2].set_title("Betw_LOW to Final_HIG")
x=between_exp_low_df["Avg"]
y=final_exp_hig_df["Avg"]

#Interpolation plot
line1, =ax[0,2].plot(x,between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,2].plot(y,final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[0,2],between_exp_low_df)
plot_feature_name_on_axe2(ax[0,2],final_exp_hig_df)

ax[0,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[1,0].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[1,0].scatter(final_exp_low_df['Avg'], final_exp_low_df['Stddev'])
ax[1,0].set_xlabel('avg')
ax[1,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,0].set_title("Betw_MID to Final_LOW")
x=between_exp_mid_df["Avg"]
y=final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[1,0].plot(x,between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,0].plot(y,final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,0],between_exp_mid_df)
plot_feature_name_on_axe2(ax[1,0],final_exp_low_df)

ax[1,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[1,1].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[1,1].scatter(final_exp_mid_df['Avg'], final_exp_mid_df['Stddev'])
ax[1,1].set_xlabel('avg')
ax[1,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,1].set_title("Betw_MID to Final_MID")
x=between_exp_mid_df["Avg"]
y=final_exp_mid_df["Avg"]

#Interpolation plot
line1, =ax[1,1].plot(x,between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,1].plot(y,final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,1],between_exp_mid_df)
plot_feature_name_on_axe2(ax[1,1],final_exp_mid_df)

ax[1,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[1,2].scatter(between_exp_mid_df['Avg'], between_exp_mid_df['Stddev'])
ax[1,2].scatter(final_exp_hig_df['Avg'], final_exp_hig_df['Stddev'])
ax[1,2].set_xlabel('avg')
ax[1,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,2].set_title("Betw_MID to Final_HIG")
x=between_exp_mid_df["Avg"]
y=final_exp_hig_df["Avg"]

#Interpolation plot
line1, =ax[1,2].plot(x,between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,2].plot(y,final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[1,2],between_exp_mid_df)
plot_feature_name_on_axe2(ax[1,2],final_exp_hig_df)

ax[1,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[2,0].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[2,0].scatter(final_exp_low_df['Avg'], final_exp_low_df['Stddev'])
ax[2,0].set_xlabel('avg')
ax[2,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,0].set_title("Betw_HIG to Final_LOW")
x=between_exp_hig_df["Avg"]
y=final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[2,0].plot(x,between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,0].plot(y,final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,0],between_exp_hig_df)
plot_feature_name_on_axe2(ax[2,0],final_exp_low_df)

ax[2,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)


ax[2,1].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[2,1].scatter(final_exp_mid_df['Avg'], final_exp_mid_df['Stddev'])
ax[2,1].set_xlabel('avg')
ax[2,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,1].set_title("Betw_HIG to Final_MID")
x=between_exp_hig_df["Avg"]
y=final_exp_mid_df["Avg"]
#Interpolation plot
line1, =ax[2,1].plot(x,between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,1].plot(y,final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,1],between_exp_hig_df)
plot_feature_name_on_axe2(ax[2,1],final_exp_mid_df)

ax[2,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)

ax[2,2].scatter(between_exp_hig_df['Avg'], between_exp_hig_df['Stddev'])
ax[2,2].scatter(final_exp_hig_df['Avg'], final_exp_hig_df['Stddev'])
ax[2,2].set_xlabel('avg')
ax[2,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,2].set_title("Betw_HIG to Final_HIG")
x=between_exp_hig_df["Avg"]
y=final_exp_hig_df["Avg"]
#Interpolation plot
line1, =ax[2,2].plot(x,between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,2].plot(y,final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
plot_feature_name_on_axe1(ax[2,2],between_exp_hig_df)
plot_feature_name_on_axe2(ax[2,2],final_exp_hig_df)

ax[2,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
between_to_final_df = between_to_final_df.append(new_element, ignore_index = True)


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(16, 10)
plt.show()

#Saving Correlation into csv
between_to_final_df.to_csv("between_to_final.csv")


# In[24]:


# BEtween to Fianal exp : Pertinant features
PC_between_to_final = {'Movement':[], 'Pearsonr_Corr':[],'Spearman_Corr':[] }
# creating a dataframe from dictionary
PC_between_to_final_df = pd.DataFrame(PC_between_to_final)

#Plot
fig, ax = plt.subplots(3,3)
ax[0,0].scatter(PC_between_exp_low_df['Avg'], PC_between_exp_low_df['Stddev'])
ax[0,0].scatter(PC_final_exp_low_df['Avg'], PC_final_exp_low_df['Stddev'])
ax[0,0].set_xlabel('avg')
ax[0,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,0].set_title("Betw_LOW to Final_LOW")
x=PC_between_exp_low_df["Avg"]
y=PC_final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[0,0].plot(x,PC_between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,0].plot(y,PC_final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[0,0],PC_between_exp_low_df)
PC_plot_feature_name_on_axe2(ax[0,0],PC_final_exp_low_df)

ax[0,0].legend(handles=[line1,line2], loc='upper right')
corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[0,1].scatter(PC_between_exp_low_df['Avg'], PC_between_exp_low_df['Stddev'])
ax[0,1].scatter(PC_final_exp_mid_df['Avg'], PC_final_exp_mid_df['Stddev'])
ax[0,1].set_xlabel('avg')
ax[0,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,1].set_title("Betw_LOW to Final_MID")
x=PC_between_exp_low_df["Avg"]
y=PC_final_exp_mid_df["Avg"]

#Interpolation plot
line1, =ax[0,1].plot(x,PC_between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,1].plot(y,PC_final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[0,1],PC_between_exp_low_df)
PC_plot_feature_name_on_axe2(ax[0,1],PC_final_exp_mid_df)

ax[0,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[0,2].scatter(PC_between_exp_low_df['Avg'], PC_between_exp_low_df['Stddev'])
ax[0,2].scatter(PC_final_exp_hig_df['Avg'], PC_final_exp_hig_df['Stddev'])
ax[0,2].set_xlabel('avg')
ax[0,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[0,2].set_title("Betw_LOW to Final_HIG")
x=PC_between_exp_low_df["Avg"]
y=PC_final_exp_hig_df["Avg"]

#Interpolation plot
line1, =ax[0,2].plot(x,PC_between_exp_low_df['Stddev'], label="Betw_LOW")
line2, =ax[0,2].plot(y,PC_final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[0,2],PC_between_exp_low_df)
PC_plot_feature_name_on_axe2(ax[0,2],PC_final_exp_hig_df)

ax[0,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Low to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[1,0].scatter(PC_between_exp_mid_df['Avg'], PC_between_exp_mid_df['Stddev'])
ax[1,0].scatter(PC_final_exp_low_df['Avg'], PC_final_exp_low_df['Stddev'])
ax[1,0].set_xlabel('avg')
ax[1,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,0].set_title("Betw_MID to Final_LOW")
x=PC_between_exp_mid_df["Avg"]
y=PC_final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[1,0].plot(x,PC_between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,0].plot(y,PC_final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[1,0],PC_between_exp_mid_df)
PC_plot_feature_name_on_axe2(ax[1,0],PC_final_exp_low_df)

ax[1,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[1,1].scatter(PC_between_exp_mid_df['Avg'], PC_between_exp_mid_df['Stddev'])
ax[1,1].scatter(PC_final_exp_mid_df['Avg'], PC_final_exp_mid_df['Stddev'])
ax[1,1].set_xlabel('avg')
ax[1,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,1].set_title("Betw_MID to Final_MID")
x=PC_between_exp_mid_df["Avg"]
y=PC_final_exp_mid_df["Avg"]

#Interpolation plot
line1, =ax[1,1].plot(x,PC_between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,1].plot(y,PC_final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[1,1],PC_between_exp_mid_df)
PC_plot_feature_name_on_axe2(ax[1,1],PC_final_exp_mid_df)

ax[1,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[1,2].scatter(PC_between_exp_mid_df['Avg'], PC_between_exp_mid_df['Stddev'])
ax[1,2].scatter(PC_final_exp_hig_df['Avg'], PC_final_exp_hig_df['Stddev'])
ax[1,2].set_xlabel('avg')
ax[1,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[1,2].set_title("Betw_MID to Final_HIG")
x=PC_between_exp_mid_df["Avg"]
y=PC_final_exp_hig_df["Avg"]

#Interpolation plot
line1, =ax[1,2].plot(x,PC_between_exp_mid_df['Stddev'], label="Betw_MID")
line2, =ax[1,2].plot(y,PC_final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[1,2],PC_between_exp_mid_df)
PC_plot_feature_name_on_axe2(ax[1,2],PC_final_exp_hig_df)

ax[1,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Mid to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[2,0].scatter(PC_between_exp_hig_df['Avg'], PC_between_exp_hig_df['Stddev'])
ax[2,0].scatter(PC_final_exp_low_df['Avg'], PC_final_exp_low_df['Stddev'])
ax[2,0].set_xlabel('avg')
ax[2,0].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,0].set_title("Betw_HIG to Final_LOW")
x=PC_between_exp_hig_df["Avg"]
y=PC_final_exp_low_df["Avg"]

#Interpolation plot
line1, =ax[2,0].plot(x,PC_between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,0].plot(y,PC_final_exp_low_df['Stddev'],label="Final_LOW")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[2,0],PC_between_exp_hig_df)
PC_plot_feature_name_on_axe2(ax[2,0],PC_final_exp_low_df)

ax[2,0].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Low" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)


ax[2,1].scatter(PC_between_exp_hig_df['Avg'], PC_between_exp_hig_df['Stddev'])
ax[2,1].scatter(PC_final_exp_mid_df['Avg'], PC_final_exp_mid_df['Stddev'])
ax[2,1].set_xlabel('avg')
ax[2,1].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,1].set_title("Betw_HIG to Final_MID")
x=PC_between_exp_hig_df["Avg"]
y=PC_final_exp_mid_df["Avg"]
#Interpolation plot
line1, =ax[2,1].plot(x,PC_between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,1].plot(y,PC_final_exp_mid_df['Stddev'],label="Final_MID")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[2,1],PC_between_exp_hig_df)
PC_plot_feature_name_on_axe2(ax[2,1],PC_final_exp_mid_df)

ax[2,1].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Mid" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)

ax[2,2].scatter(PC_between_exp_hig_df['Avg'], PC_between_exp_hig_df['Stddev'])
ax[2,2].scatter(PC_final_exp_hig_df['Avg'], PC_final_exp_hig_df['Stddev'])
ax[2,2].set_xlabel('avg')
ax[2,2].set_ylabel('stddev')
#ax[0,0].set_ylim(0,)
ax[2,2].set_title("Betw_HIG to Final_HIG")
x=PC_between_exp_hig_df["Avg"]
y=PC_final_exp_hig_df["Avg"]
#Interpolation plot
line1, =ax[2,2].plot(x,PC_between_exp_hig_df['Stddev'], label="Betw_HIG")
line2, =ax[2,2].plot(y,PC_final_exp_hig_df['Stddev'],label="Final_HIG")
#Plot of the name of the features
PC_plot_feature_name_on_axe1(ax[2,2],PC_between_exp_hig_df)
PC_plot_feature_name_on_axe2(ax[2,2],PC_final_exp_hig_df)

ax[2,2].legend(handles=[line1,line2], loc='upper right')

corrPearson, _ =pearsonr(x, y) #Pearson's correlation
corrSpearman, _ =spearmanr(x, y) #spearmanr's correlation
new_element = {'Movement':"Betw_Hig to Final_Hig" , 'Pearsonr_Corr':corrPearson , 'Spearman_Corr': corrSpearman}
PC_between_to_final_df = PC_between_to_final_df.append(new_element, ignore_index = True)


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(20, 14)
plt.show()

#Saving Correlation into csv
PC_between_to_final_df.to_csv("PC_between_to_final.csv")


# In[ ]:





# In[31]:


# BEtween to Fianal exp : Differences ploting of features avg

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

fig, ax = plt.subplots(3,3)

x= PC_between_exp_low_df['Avg']
y= PC_final_exp_low_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[0,0].bar(xlim - width/2, x, width, label='PC_between_exp_low_df')
rects2 = ax[0,0].bar(xlim + width/2, y, width, label='PC_final_exp_low_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,0].set_ylabel('Avg')
ax[0,0].set_title('Pertinent Features numerical differences')
#ax[0,0].set_xticks(xlim, labels)
#ax[0,0].set_xticklabels(labels.astype(str).values, rotation='vertical')
ax[0,0].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[0,0].legend()


x= PC_between_exp_low_df['Avg']
y= PC_final_exp_mid_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[0,1].bar(xlim - width/2, x, width, label='PC_between_exp_low_df')
rects2 = ax[0,1].bar(xlim + width/2, y, width, label='PC_final_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,1].set_ylabel('Avg')
ax[0,1].set_title('Pertinent Features numerical differences')
#ax[0,1].set_xticks(xlim, labels)
ax[0,1].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[0,1].legend()



x= PC_between_exp_low_df['Avg']
y= PC_final_exp_hig_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[0,2].bar(xlim - width/2, x, width, label='PC_between_exp_low_df')
rects2 = ax[0,2].bar(xlim + width/2, y, width, label='PC_final_exp_hig_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,2].set_ylabel('Avg')
ax[0,2].set_title('Pertinent Features numerical differences')
#ax[0,2].set_xticks(xlim, labels)
ax[0,2].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[0,2].legend()



x= PC_between_exp_mid_df['Avg']
y= PC_final_exp_low_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[1,0].bar(xlim -width/2, x, width, label='PC_between_exp_mid_df')
rects2 = ax[1,0].bar(xlim +width/2,  y, width, label='PC_final_exp_low_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,0].set_ylabel('Avg')
ax[1,0].set_title('Pertinent Features numerical differences')
#ax[1,0].set_xticks(xlim, labels)
ax[1,0].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[1,0].legend()



x= PC_between_exp_mid_df['Avg']
y= PC_final_exp_mid_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[1,1].bar(xlim -width/2, x, width, label='PC_between_exp_mid_df')
rects2 = ax[1,1].bar(xlim +width/2,  y, width, label='PC_final_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,1].set_ylabel('Avg')
ax[1,1].set_title('Pertinent Features numerical differences')
#ax[1,1].set_xticks(xlim, labels)
ax[1,1].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[1,1].legend()



x= PC_between_exp_mid_df['Avg']
y= PC_final_exp_hig_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[1,2].bar(xlim -width/2, x, width, label='PC_between_exp_mid_df')
rects2 = ax[1,2].bar(xlim +width/2, y, width, label='PC_final_exp_hig_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,2].set_ylabel('Avg')
ax[1,2].set_title('Pertinent Features numerical differences')
#ax[1,2].set_xticks(xlim, labels)
ax[1,2].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[1,2].legend()



x= PC_between_exp_hig_df['Avg']
y= PC_final_exp_low_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[2,0].bar(xlim -width/2, x, width, label='PC_between_exp_hig_df')
rects2 = ax[2,0].bar(xlim +width/2, y, width, label='PC_final_exp_low_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2,0].set_ylabel('Avg')
ax[2,0].set_title('Pertinent Features numerical differences')
#ax[2,0].set_xticks(xlim, labels)
ax[2,0].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[2,0].legend()



x= PC_between_exp_hig_df['Avg']
y= PC_final_exp_mid_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[2,1].bar(xlim -width/2, x, width, label='PC_between_exp_hig_df')
rects2 = ax[2,1].bar(xlim +width/2, y, width, label='PC_final_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2,1].set_ylabel('Avg')
ax[2,1].set_title('Pertinent Features numerical differences')
#ax[2,1].set_xticks(xlim, labels)
ax[2,1].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[2,1].legend()



x= PC_between_exp_hig_df['Avg']
y= PC_final_exp_hig_df['Avg']
labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax[2,2].bar(xlim -width/2, x, width, label='PC_between_exp_hig_df')
rects2 = ax[2,2].bar(xlim +width/2, y, width, label='PC_final_exp_hig_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2,2].set_ylabel('Avg')
ax[2,2].set_title('Pertinent Features numerical differences')
#ax[2,2].set_xticks(xlim, labels)
ax[2,2].set_xticklabels(PC_between_exp_hig_df["Feature"].astype(str).values, rotation='vertical')
ax[2,2].legend()



plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(28, 20)
plt.show()


# In[25]:


feature='branches'
full_df['7500-128-'+feature]


# In[4]:


#Shortcut

# Read the PC_final_exp_hig_df dataset
PC_final_exp_hig_df = pd.read_csv('PC_final_exp_hig.csv')

# Read the PC_final_exp_mid_df dataset
PC_final_exp_mid_df = pd.read_csv('PC_final_exp_mid.csv')

# Read the PC_final_exp_low_df dataset
PC_final_exp_low_df = pd.read_csv('PC_final_exp_low.csv')


# In[5]:


PC_final_exp_hig_df


# In[15]:


# Let's look at Pertinent features differences between the final low and mid

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

#Plot
fig, ax = plt.subplots(1,1)

x= PC_final_exp_low_df['Avg']
y= PC_final_exp_mid_df['Avg']

labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar((xlim-width/2) , x, width, label='PC_final_exp_low_df')
rects2 = ax.bar((xlim+width/2), y, width, label='PC_final_exp_mid_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Avg')
ax.set_title('Pertinent Features numerical differences in Final_{low and mid}')
ax.set_xticklabels(PC_final_exp_low_df["Feature"].astype(str).values, rotation='vertical')
ax.legend()

fig.set_size_inches(18, 10)
plt.show()


# In[16]:


# Let's look at Pertinent features differences between the final mid and hig

features =["branches","branch-load-misses","branch-misses","dTLB-stores","instructions","L1-dcache-loads","L1-dcache-stores"]

#Plot
fig, ax = plt.subplots(1,1)

x= PC_final_exp_mid_df['Avg']
y= PC_final_exp_hig_df['Avg']

labels = features
xlim = np.array([0, 1, 2, 3, 4,5,6])   # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar((xlim-width/2) , x, width, label='PC_final_exp_mid_df')
rects2 = ax.bar((xlim+width/2), y, width, label='PC_final_exp_hig_df')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Avg')
ax.set_title('Pertinent Features numerical differences in Final_{mid and hig}')
ax.set_xticklabels(PC_final_exp_mid_df["Feature"].astype(str).values, rotation='vertical')
ax.legend()

fig.set_size_inches(18, 10)
plt.show()


# In[19]:


# We trying to predict the trajectory in final-exp_{low,mid,hig}

#Variables from 16s
mayBeIn_finalLow_from_16s=0
mayBeIn_finalMid_from_16s=0
mayBeIn_finalHig_from_16s=0

#Variables from 17s
mayBeIn_finalLow_from_17s=0
mayBeIn_finalMid_from_17s=0
mayBeIn_finalHig_from_17s=0

#Variables from 18s
mayBeIn_finalLow_from_18s=0
mayBeIn_finalMid_from_18s=0
mayBeIn_finalHig_from_18s=0


# In[73]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[21]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[22]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[23]:


feature='branch-load-misses'
features_indices=1
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[24]:


feature='branch-load-misses'
features_indices=1
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[25]:


feature='branch-load-misses'
features_indices=1
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)

plt.title(feature+'_Prediction')
plt.ylabel('Avg')

plt.show()


# In[26]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[27]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(1,maxi)
    ax.set_xlim(0,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[28]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[29]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(1,maxi)
    ax.set_xlim(0,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[30]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[31]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[32]:


feature='instructions'
features_indices=4
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[33]:


feature='instructions'
features_indices=4
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[34]:


feature='instructions'
features_indices=4
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[35]:


feature='L1-dcache-loads'
features_indices=5
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(1,maxi)
    ax.set_xlim(0,30)
    ax.plot(y_vals,color='blue')

    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[36]:


feature='L1-dcache-loads'
features_indices=5
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[37]:


feature='L1-dcache-loads'
features_indices=5
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[38]:


feature='L1-dcache-stores'
features_indices=6
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[39]:


feature='L1-dcache-stores'
features_indices=6
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[40]:


feature='L1-dcache-stores'
features_indices=6
rate_and_size = '7500-128-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[52]:


#Statistics for data at 7500 Mbps with 128 packets size


#Variables from 16s
mayBeIn_finalLow_from_17s
#mayBeIn_finalMid_from_18s
#mayBeIn_finalHig_from_18s


# In[ ]:


# Test on 5000 Mbps with 1400 datasize


# In[8]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[69]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[70]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[12]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[13]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[14]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[15]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[16]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[17]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[18]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[19]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[20]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[21]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[22]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[23]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[24]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[25]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[26]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[27]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[28]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[29]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '5000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[ ]:


#Test over 2500 Mbps with 512 data size


# In[30]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[31]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[32]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[33]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[34]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[35]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[36]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[37]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[38]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[39]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[40]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[41]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[42]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[43]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[44]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[45]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[46]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[47]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[48]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_16s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_16s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_16s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[49]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_17s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_17s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_17s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[74]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '2500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalHig_from_18s+=1

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')
            #Incrementation to reckon the percentage
            mayBeIn_finalMid_from_18s+=1

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
            #Incrementation to reckon the percentage
            mayBeIn_finalLow_from_18s+=1
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[ ]:


#Test for 500-512


# In[96]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[97]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[100]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[101]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[102]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[103]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[104]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[105]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[106]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[107]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[108]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[109]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[110]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[111]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[112]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[113]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[114]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[115]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[116]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[117]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[123]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '500-512-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[ ]:


#Test of 10000-1400


# In[120]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[121]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[122]:


import matplotlib.animation as animation

feature='branches'
features_indices=0
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[124]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[125]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[126]:


import matplotlib.animation as animation

feature='branch-load-misses'
features_indices=1
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[127]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[128]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[129]:


import matplotlib.animation as animation

feature='branch-misses'
features_indices=2
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[130]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[131]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[132]:


import matplotlib.animation as animation

feature='dTLB-stores'
features_indices=3
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[133]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[134]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[135]:


import matplotlib.animation as animation

feature='instructions'
features_indices=4
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[136]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[137]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[138]:


import matplotlib.animation as animation

feature='L1-dcache-loads'
features_indices=5
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[139]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 16:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[16] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[140]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 17:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[17] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[141]:


import matplotlib.animation as animation

feature='L1-dcache-stores'
features_indices=6
rate_and_size = '10000-1400-' # rate and data siez given for the prediction
data= full_df[rate_and_size+feature]

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_vals=[]
i=0

# This function is called periodically from FuncAnimation
def animate(i):
    y_vals.append(data.iloc[i])
    maxi=data.max()
    # Draw x and y lists
    plt.cla()
    #ax.clear()
    ax.set_ylim(0,maxi)
    ax.set_xlim(1,30)
    ax.plot(y_vals,color='blue')
    
    if i>= 18:
        # between to final_exp_hig
        if PC_final_exp_hig_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_hig_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20.3, 30)
            y1= PC_final_exp_hig_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_hig_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='red')
            #ax.plot(x, (y1 + y2)/2,color='blue', linewidth=2)

        # between to final_exp_mid
        if PC_final_exp_mid_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_mid_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(20, 30)
            y1= PC_final_exp_mid_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_mid_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='blue')

        # last case: between to final_exp_low
        if PC_final_exp_low_df['CI_lower'].iloc[features_indices] < data.iloc[18] < PC_final_exp_low_df['CI_upper'].iloc[features_indices] : 
            x=np.linspace(19.7, 30)
            y1= PC_final_exp_low_df['CI_lower'].iloc[features_indices]
            y2= PC_final_exp_low_df['CI_upper'].iloc[features_indices]
            ax.fill_between(x,y1, y2, alpha=.5, color='yellow')
            #ax.axhline((y1 + y2)/2, color='b', linewidth=2)
    
    # Format plot
    #plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title(feature+'_Prediction')
    plt.ylabel('Avg')
        
    i=i+1

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)



plt.show()


# In[4]:


# Interpolation part

from scipy.interpolate import interp1d


# In[5]:


x = np.linspace(0, 31, num=29, endpoint=True)
y = full_df['10000-1400-branches']


# In[6]:


y_between = full_df['10000-1400-branches'].iloc[14:19] #Selection from 15s to the last
y_between.shape


# In[7]:


xnew = np.linspace(15, 20, num=5, endpoint=True)
xnew.shape


# In[8]:


from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
from scipy import interpolate

f1 = interp1d(xnew, y_between, kind='linear')
f2 = interp1d(xnew, y_between, kind='cubic')
f3 = interpolate.interp1d(xnew, y_between) #no precision of kind of interpolation


# In[16]:


import matplotlib.pyplot as plt
plt.plot(x, y, '-')
plt.plot(xnew, f1(xnew), '-')
plt.plot(xnew, f2(xnew), '--')
plt.plot(xnew, f3(xnew), '-.')
#plt.plot(xnew, Polynomial(poly.coef[::-1])(xnew))
plt.legend(['original', 'linear', 'cubic','interp1d'], loc='best')
plt.show()


# In[10]:


#prediction of point in future 
print(f1(20))


# In[ ]:


#Trying to have the best Linear extrapolation


# In[17]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 20th output
yLinear = ((f1(20)-f1(15))/5) * x +  ( f1(20)- (20*((f1(20)-f1(15))/5)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(28, yLinear[28], 'ro')
plt.show()


# In[18]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 18.9th output
yLinear = ((f1(18.9)-f1(15))/3.9) * x +  ( f1(18.9)- (18.9*((f1(18.9)-f1(15))/3.9)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(15, f1(15), 'ro')


# In[19]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 18.9th output
yLinear = ((f1(19)-f1(15))/4) * x +  ( f1(19)- (19*((f1(19)-f1(15))/4)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(15, f1(15), 'ro')


# In[20]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 18.9th output
yLinear = ((f1(19.1)-f1(15))/4.1) * x +  ( f1(19.1)- (19.1*((f1(19.1)-f1(15))/4.1)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(15, f1(15), 'ro')


# In[21]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 18.9th output
yLinear = ((f1(19.3)-f1(15))/4.3) * x +  ( f1(19.3)- (19.3*((f1(19.3)-f1(15))/4.3)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(15, f1(15), 'ro')


# In[22]:


fig = plt.figure()
plt.plot(x, y, '-')

#Linear interpolation passing through two points the 15th and 18.9th output
yLinear = ((f1(19.2)-f1(15))/4.2) * x +  ( f1(19.2)- (19.2*((f1(19.2)-f1(15))/4.2)))
plt.plot(x, yLinear, '-r')
plt.legend(['original','linear_extrapolation'], loc='best')

#add estimated y-value to plot
plt.plot(15, f1(15), 'ro')


# In[23]:


from itertools import repeat
fig = plt.figure()
plt.plot(x, y, '-')

xFinalpart = np.linspace(15, 30, num=15, endpoint=True)

yFinalpart=[]
for i in range(15,20):
  yFinalpart.append(f3(i)) #basic interpolation in the change part

#Addition of the final part based on the twentieth value
for i in range(0,10):
  yFinalpart.append(f3(20))

plt.plot(xFinalpart,yFinalpart)

plt.legend(['original','basic_extrapolation'], loc='best')


# In[24]:


fig = plt.figure()
from itertools import repeat
plt.plot(x, y, '-')

xFinalpart = np.linspace(15, 30, num=15, endpoint=True)

yFinalpart=[]
for i in range(15,20):
  yFinalpart.append(f2(i)) #cubic interpolation in the change part

#Addition of the final part based on the twentieth value
for i in range(0,10):
  yFinalpart.append(f2(20))

plt.plot(xFinalpart,yFinalpart)

plt.legend(['original','linear_extrapolation'], loc='best')


# In[25]:


from itertools import repeat
fig = plt.figure()
plt.plot(x, y, '-')

xFinalpart = np.linspace(15, 30, num=15, endpoint=True)

yFinalpart=[]
for i in range(15,20):
  yFinalpart.append(f1(i)) #linear interpolation in the change part

#Addition of the final part based on the twentieth value
for i in range(0,10):
  yFinalpart.append(f1(20))

plt.plot(xFinalpart,yFinalpart)

plt.legend(['original','cubic_extrapolation'], loc='best')


# In[155]:


#PCA 

#PCA full_df

data = full_df.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

# We separate the last column in 
train_x, train_y = np.hsplit(train, [train.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int)

test_x, test_y = np.hsplit(test, [test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

# We apply Min-Max scaler. In this code we do it by hand, but we could have done it also by directly using sklearn.preprocessing.MinMaxScaler.

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)


#For PCA part

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component
explained_variance = pca.explained_variance_ratio_
#explained_variance.sort()

explained_variance

#It can be seen that first principal component is responsible for 63.30% variance.Similarly, the second principal 
#component causes 11.32% variance in the dataset. Collectively we can say that (63.30 + 11.32) 74,62% percent of 
#the classification information contained in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm.
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


#Training and Making Predictions
#In this case we'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Performance Evaluation
cm= confusion_matrix(test_y, y_pred)
print(cm)

print("The accuracy is ", accuracy_score(test_y, y_pred) )

projected = pca.fit_transform(data)
print(data.shape)
print(projected.shape)



# In[93]:


import matplotlib.pyplot as plt
import numpy as np

# make data
np.random.seed(1)
x = np.linspace(21, 30)
y1 = PC_final_exp_hig_df['Avg'].iloc[features_indices]- PC_final_exp_hig_df['Stddev'].iloc[features_indices]
y2 = PC_final_exp_hig_df['Avg'].iloc[features_indices]+ PC_final_exp_hig_df['Stddev'].iloc[features_indices]

# plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=1,color='r')


plt.show()

y1 = PC_final_exp_hig_df['Avg'].iloc[features_indices: 29]


# In[ ]:


# between to final_exp_mid
elif ( PC_final_exp_mid_df['Avg'].iloc[features_indices] - (1+ PC_final_exp_mid_df['Stddev'].iloc[features_indices])) < data.iloc[i] < (PC_final_exp_mid_df['Avg'].iloc[features_indices] + (1+ PC_final_exp_mid_df['Stddev'].iloc[features_indices])): 
    x=PC_final_exp_mid_df['Avg'].iloc[features_indices]
    y1= PC_final_exp_mid_df['Avg'].iloc[features_indices]- PC_final_exp_mid_df['Stddev'].iloc[features_indices]
    y2= PC_final_exp_mid_df['Avg'].iloc[features_indices]+ PC_final_exp_mid_df['Stddev'].iloc[features_indices]
    ax.fill_between(x,y1, y2, alpha=.5, linewidth=0)
    ax.plot(x, (y1 + y2)/2, linewidth=2)
              
# last case: between to final_exp_low
elif ( PC_final_exp_low_df['Avg'].iloc[features_indices] - (1+ PC_final_exp_low_df['Stddev'].iloc[features_indices])) < data.iloc[i] < (PC_final_exp_low_df['Avg'].iloc[features_indices] + (1+ PC_final_exp_low_df['Stddev'].iloc[features_indices])): 
    x=PC_final_exp_low_df['Avg'].iloc[features_indices]
    y1= PC_final_exp_low_df['Avg'].iloc[features_indices]- PC_final_exp_low_df['Stddev'].iloc[features_indices]
    y2= PC_final_exp_low_df['Avg'].iloc[features_indices]+ PC_final_exp_low_df['Stddev'].iloc[features_indices]
    ax.fill_between(x,y1, y2, alpha=.5, linewidth=0)
    ax.plot(x, (y1 + y2)/2, linewidth=2)
    


# In[50]:


#Plot of the Correlation Spearman_Corr

def plot_feature_name_of_corr(ax,exp_name):
    i=0
    for i in range(0,3):
        ax.annotate(exp_name['Movement'].iloc[i], xy =(i,exp_name['Movement'].iloc[i]),
             xytext =(i,exp_name['Movement'].iloc[i]),
             arrowprops = dict(facecolor ='red',
                                shrink = 0.01),   )
        i=i+1

#Plot
fig, ax = plt.subplots(2,3)


ax[0,0].plot(zero_to_between_df['Spearman_Corr'].iloc[0:3,], 'go')
ax[0,0].plot(zero_to_between_df['Spearman_Corr'].iloc[0:3,], color="green")
#plot_feature_name_of_corr(ax[0,0],zero_to_between_df)
ax[0,0].set_xlabel('Mvt')
ax[0,0].set_ylabel('Spearman_Corr')
ax[0,0].set_title("Z_L to B_{L/M/H}")

ax[0,1].plot(zero_to_between_df['Spearman_Corr'].iloc[3:6,], 'bo')
ax[0,1].plot(zero_to_between_df['Spearman_Corr'].iloc[3:6,], color="b")
#plot_feature_name_of_corr(ax[0,1],zero_to_between_df)
ax[0,1].set_xlabel('Mvt')
ax[0,1].set_ylabel('Spearman_Corr')
ax[0,1].set_title("Z_M to B_{L/M/H}")

ax[0,2].plot(zero_to_between_df['Spearman_Corr'].iloc[6:9,], 'ro')
ax[0,2].plot(zero_to_between_df['Spearman_Corr'].iloc[6:9,], color="r")
#plot_feature_name_of_corr(ax[0,2],zero_to_between_df)
ax[0,2].set_xlabel('Mvt')
ax[0,2].set_ylabel('Spearman_Corr')
ax[0,2].set_title("Z_H to B_{L/M/H}")


ax[1,0].plot(between_to_final_df['Spearman_Corr'].iloc[0:3,], 'go')
ax[1,0].plot(between_to_final_df['Spearman_Corr'].iloc[0:3,], color="green")
#plot_feature_name_of_corr(ax[1,0],between_to_final_df)
ax[1,0].set_xlabel('Mvt')
ax[1,0].set_ylabel('Spearman_Corr')
ax[1,0].set_title("B_L to F_{L/M/H}")

ax[1,1].plot(between_to_final_df['Spearman_Corr'].iloc[3:6,], 'bo')
ax[1,1].plot(between_to_final_df['Spearman_Corr'].iloc[3:6,], color="b")
#plot_feature_name_of_corr(ax[1,1],between_to_final_df)
ax[1,1].set_xlabel('Mvt')
ax[1,1].set_ylabel('Spearman_Corr')
ax[1,1].set_title("B_M to F_{L/M/H}")

ax[1,2].plot(between_to_final_df['Spearman_Corr'].iloc[6:9,], 'bo')
ax[1,2].plot(between_to_final_df['Spearman_Corr'].iloc[6:9,], color="b")
#plot_feature_name_of_corr(ax[1,2],between_to_final_df)
ax[1,2].set_xlabel('Mvt')
ax[1,2].set_ylabel('Spearman_Corr')
ax[1,2].set_title("B_H to F_{L/M/H}")



plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(9, 6)
plt.show()


# In[601]:


#Plot of the Correlation

#Plot
fig, ax = plt.subplots(2,3)
Pearsonr_Corr='Pearsonr_Corr'
ax[0,0].plot(zero_to_between_df['Pearsonr_Corr'].iloc[0:3,], 'go')
ax[0,0].plot(zero_to_between_df['Pearsonr_Corr'].iloc[0:3,], color="green")
ax[0,0].set_xlabel('Mvt')
ax[0,0].set_ylabel('Pearsonr_Corr')
ax[0,0].set_title("Z_L to B_{L/M/H}")
#plot_feature_name_of_corr(ax[0,0],zero_to_between_df)
ax[0,1].plot(zero_to_between_df['Pearsonr_Corr'].iloc[3:6,], 'bo')
ax[0,1].plot(zero_to_between_df['Pearsonr_Corr'].iloc[3:6,], color="b")
ax[0,1].set_xlabel('Mvt')
ax[0,1].set_ylabel('Pearsonr_Corr')
ax[0,1].set_title("Z_M to B_{L/M/H}")
#plot_feature_name_of_corr(ax[0,1],zero_to_between_df)
ax[0,2].plot(zero_to_between_df['Pearsonr_Corr'].iloc[6:9,], 'ro')
ax[0,2].plot(zero_to_between_df['Pearsonr_Corr'].iloc[6:9,], color="r")
ax[0,2].set_xlabel('Mvt')
ax[0,2].set_ylabel('Pearsonr_Corr')
ax[0,2].set_title("Z_H to B_{L/M/H}")
#plot_feature_name_of_corr(ax[0,2],zero_to_between_df)



ax[1,0].plot(between_to_final_df['Pearsonr_Corr'].iloc[0:3,], 'go')
ax[1,0].plot(between_to_final_df['Pearsonr_Corr'].iloc[0:3,], color="green")
ax[1,0].set_xlabel('Mvt')
ax[1,0].set_ylabel('Pearsonr_Corr')
ax[1,0].set_title("B_L to F_{L/M/H}")
#plot_feature_name_of_corr(ax[1,0],zero_to_between_df)
ax[1,1].plot(between_to_final_df['Pearsonr_Corr'].iloc[3:6,], 'bo')
ax[1,1].plot(between_to_final_df['Pearsonr_Corr'].iloc[3:6,], color="b")
ax[1,1].set_xlabel('Mvt')
ax[1,1].set_ylabel('Pearsonr_Corr')
ax[1,1].set_title("B_M to F_{L/M/H}")
#plot_feature_name_of_corr(ax[1,1],zero_to_between_df)
ax[1,2].plot(between_to_final_df['Pearsonr_Corr'].iloc[6:9,], 'bo')
ax[1,2].plot(between_to_final_df['Pearsonr_Corr'].iloc[6:9,], color="b")
ax[1,2].set_xlabel('Mvt')
ax[1,2].set_ylabel('Pearsonr_Corr')
ax[1,2].set_title("B_H to F_{L/M/H}")
#plot_feature_name_of_corr(ax[1,2],zero_to_between_df)


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.4) 
fig.set_size_inches(9, 6)
plt.show()


# In[620]:


def make_sequential_model(sample_size, layer_sizes, 
                          hidden_activation_function="sigmoid", 
                          out_activation_function="softmax",
                          loss_fun="sparse_categorical_crossentropy",
                          learning_rate=0.01,
                          regularization_coeff=0,
                          metrics=['accuracy']):
  """
  Makes a sequential model.
  Parameters
  -------------------------
  sample_size: integer
    The number of features of the samples

  layer_sizes: list
    List of the size of the neural network layers. For instance, if
    layer_sizes = [8, 6, 4], the 1st layer will have 5 neurons, the 2nd 6 etc.
    Attention: the size of the last layer (the output layer) is not arbitrary.
    In case of monodimensional regression, it must be 1.
    When using categorical_crossentropy, it must be the same as the number of 
    categories.
    When using binary_crossentropy, it must be 1.

  inner_activation_function: string
    Activation function used in all layers, except the last one.
    Ex: "relu"

  out_activation_function: string
  Activation function of the last layer.
    Ex. "softmax"

  loss_fun: string
    The loss function we want to minimize. Ex. categorical_crossentropy

  learning_rate: float
    Ex. 0.01

  regularization_coeff: float
    Coefficient of ridge regression
    Ex. 0.01

  metrics: list of strings
    The metrics we want to show during training. Ex. ['accuracy']
  """

  model = Sequential()


  # In the next code we will use `partial`, which is a function of the ptyhon
  # library functools, which allows to define a class, identical to another
  # class but with some different default values.
  # In our case we define MyDenseLayer equal to the standard keras class
  # `Dense`, which implements a simple neural network layer, specifying 
  # two default values: one for the activation function, and another for the
  # regularization

  if (regularization_coeff==0):
    # No regularization
    MyDenseLayer = partial(Dense, activation=hidden_activation_function)
  else:
    MyDenseLayer = partial(Dense, activation=hidden_activation_function,
                         kernel_regularizer=keras.regularizers.l2(regularization_coeff))

  # Add the input layer
  model.add( MyDenseLayer(layer_sizes[0], 
                  input_dim = sample_size) )
  
  # Add hidden layers
  for i in range(1,len(layer_sizes)-1 ): # We iterate from the 2nd element to the penultimate
    model.add( MyDenseLayer(layer_sizes[i]) )
    
  # Add output layer
  model.add( Dense(layer_sizes[-1],
                  activation = out_activation_function) )
  

  model.compile(loss=loss_fun, 
              optimizer=keras.optimizers.Adam(lr=learning_rate) ,
              metrics=metrics)
  
  return model


def enforce_reproducibility(seed):
  tf.keras.backend.clear_session()

  # To know more: 
  #       https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
  random.seed(seed)
  np.random.seed(random.randint(0,300000))
  tf.random.set_seed(random.randint(0,300000))

def train_model(model, nn_file, X_tr, y_tr, seed, max_epochs=1000, 
                overwrite=True, validation_split=0.2, patience=20):
  """
  model: neural network model
       It must be a compiled neural network, e.g., a model issued by the
       function make_sequential_model(..) defined before

  nn_file: string (name of a file)
         This file will be used to store the weights of the trained neural
         network. Such weights are automatically stored during training 
         (thanks to the ModelCheckpoint callback (see the implementation 
         code)), so that even if the code fails in the middle of training,
         you can resume training without starting from scratch.
         If the file already exists, before starting training, the weights
         in such a file will be loaded, so that we do not start training from
         scratch, but we start already from (hopefully) good weigths.
  
  overwrite: boolean
           If true, the model will be built and trained from scratch, 
           indipendent of whether nn_file exists or not.

  seed: integer

  X_tr: matrix
      Feature matrix of the training set

  y_tr: matrix
      True labels of the training set

  max_epochs: integer
            Training will stop after such number of epochs

  validation_split: float (between 0 and 1)
                 Fraction of training dataset that will be used as validation

  patience: integer
          Training will stop if the validation loss does not improve after the 
          specified number of epochs
  """
  
  enforce_reproducibility(seed)


  # Before starting training, Keras divides (X_tr, y_tr) into a training subset
  # and a validation subset. During iterations, Keras will do backpropagation
  # in order to minimize the loss on the trainins subset, but it will monitor 
  # and also plot the loss on the validation subset.
  # However, Keras always takes the first part of (X_tr, y_tr) as training
  # subset and the second part as validation subset. This can be bad, in case
  # the dataset has been created with a certain order (for instance all the 
  # samples with a certain characteristic first, and then all the others), as
  # we instead need to train the neural network on a representative subset of 
  # samples. For this reason, we first shuffle the dataset
  X_train, y_train = shuffle(X_tr, y_tr, random_state=seed)


  ##################
  #### CALLBACKS ###
  ##################
  # These functions are called at every epoch
  plot_cb = PlotLossesKerasTF()  # Plots the loss
  checkpoint_cb = ModelCheckpoint(nn_file) # Stores weights
  logger_cb = CSVLogger(nn_file+'.csv', append=True) # Stores history
                # see https://theailearner.com/2019/07/23/keras-callbacks-csvlogger/


  # To stop early if we already converged
  # See pagg 315-16 of [Ge19]
  early_stop_cb = tf.keras.callbacks.EarlyStopping(verbose=1,
      monitor='val_loss',
     patience=patience, restore_best_weights=True)
  
  if overwrite==True:
    try:
      os.remove(nn_file)
    except OSError:
      pass

    try:
      os.remove(nn_file+'.csv')
    except OSError:
      pass

  if isfile(nn_file):
    print("Loading pre-existing model")
    model = load_model(nn_file)

  history = model.fit(X_train, y_train, epochs=max_epochs, 
                      callbacks = [plot_cb, checkpoint_cb, logger_cb, early_stop_cb], 
                      validation_split=validation_split )

  return history


# In[1]:


# Classes definition base on rate

low_class = pd.concat([data_500_rate, data_2500_rate],axis=1)
mid_class = data_5000_rate
hig_class = pd.concat([data_7500_rate, data_10000_rate],axis=1)

full= pd.concat([data_500_rate, data_2500_rate,data_5000_rate,data_7500_rate, data_10000_rate],axis=1)


# In[740]:


# For full shape
data= full
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=1 )

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train)
standardized_X = scaler.transform(train)
standardized_X_test = scaler.transform(test)

# We separate the last column in 
train_x, train_y = np.hsplit(standardized_X, [standardized_X.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int) 

test_x, test_y = np.hsplit(standardized_X_test, [standardized_X_test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)



# Logistic Regression on the full Data

#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is faster for large ones.
#For multiclass problems, only ‘newton-cg’ and ‘lbfgs’ handle multinomial loss; ‘sag’ and ‘liblinear’ are limited to one-versus-rest schemes.

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="ovr", solver="sag", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=200)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))


from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


from sklearn import neighbors
model =  neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) ) 


#For neural network
nn_file = "./"
sample_size = train_x.shape[1]
num_of_classes=len(class_names)
shallow_architecture=[20, 10, 5, num_of_classes]
model = make_sequential_model(sample_size, shallow_architecture)
history = train_model(model, nn_file, train_x, train_y, seed=3)


# In[735]:


#PCA full_df

data = full_df.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

# We separate the last column in 
train_x, train_y = np.hsplit(train, [train.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int)

test_x, test_y = np.hsplit(test, [test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

# We apply Min-Max scaler. In this code we do it by hand, but we could have done it also by directly using sklearn.preprocessing.MinMaxScaler.

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)


#For PCA part

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component
explained_variance = pca.explained_variance_ratio_
#explained_variance.sort()

explained_variance

#It can be seen that first principal component is responsible for 63.30% variance.Similarly, the second principal 
#component causes 11.32% variance in the dataset. Collectively we can say that (63.30 + 11.32) 74,62% percent of 
#the classification information contained in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm.
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


#Training and Making Predictions
#In this case we'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Performance Evaluation
cm= confusion_matrix(test_y, y_pred)
print(cm)

print("The accuracy is ", accuracy_score(test_y, y_pred) )

projected = pca.fit_transform(data)
print(data.shape)
print(projected.shape)


###############################################
#I have to review the PCA with n_components=2 #


# In[751]:


# For low shape
data= low_class
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=1 )

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train)
standardized_X = scaler.transform(train)
standardized_X_test = scaler.transform(test)

# We separate the last column in 
train_x, train_y = np.hsplit(standardized_X, [standardized_X.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int) 

test_x, test_y = np.hsplit(standardized_X_test, [standardized_X_test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)



# Logistic Regression on the full Data

#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is faster for large ones.
#For multiclass problems, only ‘newton-cg’ and ‘lbfgs’ handle multinomial loss; ‘sag’ and ‘liblinear’ are limited to one-versus-rest schemes.

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="ovr", solver="sag", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=200)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))


from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


from sklearn import neighbors
model =  neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) ) 


# In[749]:


#PCA low_class

data = low_class.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

# We separate the last column in 
train_x, train_y = np.hsplit(train, [train.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int)

test_x, test_y = np.hsplit(test, [test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

# We apply Min-Max scaler. In this code we do it by hand, but we could have done it also by directly using sklearn.preprocessing.MinMaxScaler.

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)


#For PCA part

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component
explained_variance = pca.explained_variance_ratio_
#explained_variance.sort()

explained_variance

#It can be seen that first principal component is responsible for 63.30% variance.Similarly, the second principal 
#component causes 11.32% variance in the dataset. Collectively we can say that (63.30 + 11.32) 74,62% percent of 
#the classification information contained in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm.
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


#Training and Making Predictions
#In this case we'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Performance Evaluation
cm= confusion_matrix(test_y, y_pred)
print(cm)

print("The accuracy is ", accuracy_score(test_y, y_pred) )

projected = pca.fit_transform(data)
print(data.shape)
print(projected.shape)


###############################################
#I have to review the PCA with n_components=2 #


# In[754]:


# For mid shape
data= mid_class
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=1 )

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train)
standardized_X = scaler.transform(train)
standardized_X_test = scaler.transform(test)

# We separate the last column in 
train_x, train_y = np.hsplit(standardized_X, [standardized_X.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int) 

test_x, test_y = np.hsplit(standardized_X_test, [standardized_X_test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)



# Logistic Regression on the full Data

#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is faster for large ones.
#For multiclass problems, only ‘newton-cg’ and ‘lbfgs’ handle multinomial loss; ‘sag’ and ‘liblinear’ are limited to one-versus-rest schemes.

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="ovr", solver="sag", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=200)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))


from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
#plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


from sklearn import neighbors
model =  neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) ) 


# In[755]:


#PCA mid_class

data = mid_class.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

# We separate the last column in 
train_x, train_y = np.hsplit(train, [train.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int)

test_x, test_y = np.hsplit(test, [test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

# We apply Min-Max scaler. In this code we do it by hand, but we could have done it also by directly using sklearn.preprocessing.MinMaxScaler.

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)


#For PCA part

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component
explained_variance = pca.explained_variance_ratio_
#explained_variance.sort()

explained_variance

#It can be seen that first principal component is responsible for 63.30% variance.Similarly, the second principal 
#component causes 11.32% variance in the dataset. Collectively we can say that (63.30 + 11.32) 74,62% percent of 
#the classification information contained in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm.
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


#Training and Making Predictions
#In this case we'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Performance Evaluation
cm= confusion_matrix(test_y, y_pred)
print(cm)

print("The accuracy is ", accuracy_score(test_y, y_pred) )

projected = pca.fit_transform(data)
print(data.shape)
print(projected.shape)


###############################################
#I have to review the PCA with n_components=2 #


# In[756]:


# For hig shape
data= hig_class
train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=1 )

#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train)
standardized_X = scaler.transform(train)
standardized_X_test = scaler.transform(test)

# We separate the last column in 
train_x, train_y = np.hsplit(standardized_X, [standardized_X.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int) 

test_x, test_y = np.hsplit(standardized_X_test, [standardized_X_test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)



# Logistic Regression on the full Data

#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is faster for large ones.
#For multiclass problems, only ‘newton-cg’ and ‘lbfgs’ handle multinomial loss; ‘sag’ and ‘liblinear’ are limited to one-versus-rest schemes.

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="ovr", solver="sag", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=600)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )

#multi_class="multinomial", solver="newton-cg" : not good result
model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=200)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))


from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) )


from sklearn import neighbors
model =  neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
class_names = np.array(["low", "mid", "high"] )
plot_conf_mat(test_y, y_pred, class_names)
print("The accuracy is ", accuracy_score(test_y, y_pred) ) 


# In[757]:


#PCA hig_class

data = hig_class.to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=0)

# We separate the last column in 
train_x, train_y = np.hsplit(train, [train.shape[1] - 1])
train_y = train_y.reshape(-1).astype(int)

test_x, test_y = np.hsplit(test, [test.shape[1] - 1])
test_y = test_y.reshape(-1).astype(int)

# We apply Min-Max scaler. In this code we do it by hand, but we could have done it also by directly using sklearn.preprocessing.MinMaxScaler.

min = train_x.min(axis=0)
max = train_x.max(axis=0)

train_x = (train_x - min)/(max - min)
test_x = (test_x - min)/(max - min)


#For PCA part

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#The explained_variance variable is now a float type array which contains variance ratios for each principal component
explained_variance = pca.explained_variance_ratio_
#explained_variance.sort()

explained_variance

#It can be seen that first principal component is responsible for 63.30% variance.Similarly, the second principal 
#component causes 11.32% variance in the dataset. Collectively we can say that (63.30 + 11.32) 74,62% percent of 
#the classification information contained in the feature set is captured by the first two principal components.

#Let's first try to use 1 principal component to train our algorithm.
pca = PCA(n_components=1)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


#Training and Making Predictions
#In this case we'll use random forest classification for making the predictions.

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Performance Evaluation
cm= confusion_matrix(test_y, y_pred)
print(cm)

print("The accuracy is ", accuracy_score(test_y, y_pred) )

projected = pca.fit_transform(data)
print(data.shape)
print(projected.shape)


###############################################
#I have to review the PCA with n_components=2 #

