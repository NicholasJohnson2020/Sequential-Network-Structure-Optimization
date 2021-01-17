#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle


# In[6]:


data = {}
with open('Experiment_Results/num_agents_10trials_3.pickle', 'rb') as handle:
    data['num_agents'] = pickle.load(handle)
with open('Experiment_Results/T_10trials_3.pickle', 'rb') as handle:
    data['T'] = pickle.load(handle)
with open('Experiment_Results/m_10trials_3.pickle', 'rb') as handle:
    data['m'] = pickle.load(handle)


# In[74]:


plt.rcParams["font.family"] = "Times New Roman"
policies = ['Control',
            'Perpetual Random',
            'Initial Random',
            'Myopic',
            'One Step Lookahead',
            'Modifed Reconnect']

marker_dict = {'Control': 'o',
               'Perpetual Random': 'v',
               'Initial Random': '^',
               'Myopic': 's',
               'One Step Lookahead': 'd',
               'Modifed Reconnect': 'X'}

normalized_plots = [['Cumulative','num_agents','Objective'],
                    #['Cumulative','num_agents','Time'],
                    ['Cumulative','T','Objective'],
                    #['Cumulative','T','Time'],
                    ['Terminal','num_agents','Objective'],
                    #['Terminal','num_agents','Time'],
                    #['Terminal','T','Time']
                   ]

x_labels = {'num_agents': 'Number of Nodes |V|',
           'T': 'Time Horizon T',
           'm': 'Number of Edges Formed by Entering\nNodes during Network Construction'}
y_labels = {'num_agents': '\ndivided by |V|',
            'T': '\ndivided by T'}


# In[83]:


def generate_subplot(ax, value, param, data, obj_mode,
                     exog_mode, policies, include_title):

    normalized = [obj_mode,param,value] in normalized_plots
    mean_label = 'Mean ' + value
    variables = data[param]['Parameters'][param]
    hyper_params = data[param]['Parameters'].copy()
    del(hyper_params[param])

    for policy in policies:

        policy_data = data[param][obj_mode][exog_mode][policy]
        mean_data = np.zeros(len(variables))
        if normalized:
            for i in range(len(variables)):
                mean_data[i] = policy_data[variables[i]][mean_label]/variables[i]
        else:
            for i in range(len(variables)):
                mean_data[i] = policy_data[variables[i]][mean_label]

        if policy == 'Modifed Reconnect':
            label = 'Gradient Based'
        else:
            label = policy
        ax.plot(variables, mean_data, linewidth=2.5, label=label, marker=marker_dict[policy], markersize=16)

    #ax.legend()
    ax.set_xlabel(x_labels[param], fontsize=24)

    if value == 'Objective':
        y_label = 'Average Objective Value'
    else:
        y_label = 'Average Execution Time'
    if normalized:
        y_label = y_label + y_labels[param]
    ax.set_ylabel(y_label, fontsize=24)
    ax.tick_params(labelsize=20)
    if include_title:
        if value == 'Objective':
            title = 'Average Objective Value'
        else:
            title = 'Average Execution Time (seconds)'
        ax.set_title(title, fontsize=24, pad=40)

    ax.grid()


# In[84]:


def generate_plots(params, data, obj_mode,
                   exog_mode, policies, figsize,
                   filename=None):

    n = len(params)
    fig, ax = plt.subplots(n, 2, figsize=figsize)

    for i in range(n):
        if i == 0:
            title = True
        else:
            title = False
        generate_subplot(ax=ax[i,0],value='Objective', param=params[i],
                         data=data, obj_mode=obj_mode, exog_mode=exog_mode,
                         policies=policies, include_title = title)
        generate_subplot(ax=ax[i,1],value='Time', param=params[i],
                         data=data, obj_mode=obj_mode, exog_mode=exog_mode,
                         policies=policies, include_title = title)

    handles, labels = ax[n-1,0].get_legend_handles_labels()

    fig.subplots_adjust(bottom=0.15, wspace = 0.3, hspace=0.25)
    leg = fig.legend(handles, labels, loc='lower center',
                     fancybox=True, shadow=True, ncol=2, fontsize=24)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5)

    if filename != None:
        plt.savefig(filename, dpi = 300)
    plt.show()


# In[85]:


params = ['num_agents', 'T', 'm']
generate_plots(params=params, data=data, obj_mode='Cumulative',
               exog_mode='Weighted', policies=policies,
               figsize=(16,23),
               filename='Experiment_Results/test_Cumulative_Weighted_6Policies')


# In[86]:


def generate_plots_ijcai(params, data, obj_mode,
                         exog_mode, policies, figsize,
                         mode='Objective', filename=None):

    assert mode in ['Objective', 'Time']

    n = len(params)
    fig, ax = plt.subplots(1, n, figsize=figsize)

    for i in range(n):
        generate_subplot(ax=ax[i],value=mode, param=params[i],
                         data=data, obj_mode=obj_mode, exog_mode=exog_mode,
                         policies=policies, include_title=False)

    handles, labels = ax[n - 1].get_legend_handles_labels()

    fig.subplots_adjust(bottom=0.15, wspace = 0.3)
    leg = fig.legend(handles, labels, loc='upper center',
                     fancybox=True, shadow=True, ncol=6, fontsize=24)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5)

    if filename != None:
        plt.savefig(filename, dpi = 300)
    plt.show()


# In[94]:


params = ['num_agents', 'T', 'm']
generate_plots_ijcai(params=params, data=data, obj_mode='Terminal',
                     exog_mode='Weighted', policies=policies,
                     figsize=(23, 8), mode='Time',
                     filename='Experiment_Results/Terminal_Weighted_Time_ijcai')


# In[ ]:
