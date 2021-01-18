import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import pickle

# Verify the number of command line arguments
assert len(sys.argv) == 5

num_agents_path = sys.argv[1]
T_path = sys.argv[2]
m_path = sys.argv[3]
output_path_root = sys.argv[4]

data = {}
with open(num_agents_path, 'rb') as handle:
    data['num_agents'] = pickle.load(handle)
with open(T_path, 'rb') as handle:
    data['T'] = pickle.load(handle)
with open(m_path, 'rb') as handle:
    data['m'] = pickle.load(handle)

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
                    ['Cumulative','T','Objective'],
                    ['Terminal','num_agents','Objective']
                   ]

x_labels = {'num_agents': 'Number of Nodes |V|',
           'T': 'Time Horizon T',
           'm': 'Number of Edges Formed by Entering\nNodes during Network Construction'}
y_labels = {'num_agents': '\ndivided by |V|',
            'T': '\ndivided by T'}

def generate_subplot(ax, value, param, data, obj_mode,
                     exog_mode, policies, include_title):

    normalized = [obj_mode, param, value] in normalized_plots
    mean_label = 'Mean ' + value
    variables = data[param]['Parameters'][param]
    hyper_params = data[param]['Parameters'].copy()
    del(hyper_params[param])

    for policy in policies:

        policy_data = data[param][obj_mode][exog_mode][policy]
        mean_data = np.zeros(len(variables))
        if normalized:
            for i in range(len(variables)):
                mean_data[i] = policy_data[variables[i]][
                                            mean_label] / variables[i]
        else:
            for i in range(len(variables)):
                mean_data[i] = policy_data[variables[i]][mean_label]

        if policy == 'Modifed Reconnect':
            label = 'Gradient Based'
        else:
            label = policy
        ax.plot(variables, mean_data, linewidth=2.5,
                label=label, marker=marker_dict[policy], markersize=16)

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

params = ['num_agents', 'T', 'm']
obj_modes = ['Cumulative', 'Terminal']
exog_modes = ['Uniform', 'Weighted']
plot_modes = ['Objective', 'Time']

for obj_mode in obj_modes:
    for exog_mode in exog_modes:
        for plot_mode in plot_modes:
            output_path = output_path_root + obj_mode + '_' + exog_mode + \
                            '_' + plot_mode + '_plots'
            generate_plots_ijcai(params=params, data=data, obj_mode=obj_mode,
                                 exog_mode=exog_mode, policies=policies,
                                 figsize=(23, 8), mode=plot_mode,
                                 filename=output_path)
