#!/usr/bin/python
"""
Plotting on a large number of facets
====================================

_thumb: .4, .3

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#sns.set(style="ticks")

mapping = {'lulesh' : 2, 'cg' : 4, 'jacobi' : 5}
procs = {'lulesh' : 9, 'cg' : 16, 'jacobi' : 20}
for j in ['lulesh','cg','jacobi']: 
    for i in ['std','ms','fs','global']:
        #sns.set()
        path = 'data/'+j+'/stats-'+i
        my_data = pd.read_csv(path+'.csv',comment='#')
        my_data['Joules'] -= 7.
        my_data['Joules'] /= mapping.get(j)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        g = sns.barplot(x="Iteration",y="Duration",hue="Rank",data=my_data, palette="Set3", ax=ax1)
        g.legend_.remove()
        sns.despine(ax=ax1,left='True')
        g = sns.barplot(x="Iteration",y="Joules",hue="Rank",data=my_data, palette="Set3", ax=ax2)
        g.set_yscale('log')
        #g.legend_.remove()
        fig.savefig(path+'.png',bbox_inches='tight')

for j in ['lulesh','cg','jacobi']: 
    fig = plt.figure()
    for i in ['std','ms','fs','global']:
        path = 'data/'+j+'/stats-'+i
        my_data = pd.read_csv(path+'.csv',comment='#')
        original_joules = my_data['Joules']
        original_joules = original_joules[::mapping.get(j)]
        original_iters = my_data['Iteration']
        original_iters = original_iters[::mapping.get(j)]
        g2 = sns.lineplot(x=original_iters,y=original_joules, palette="Set3")
    fig.savefig('data/'+j+'/merged.png',bbox_inches='tight')
