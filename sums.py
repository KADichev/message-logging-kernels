#!/home/kiril/anaconda3/bin/python3.7

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mapping = {'lulesh' : 2, 'cg' : 4, 'jacobi' : 5}
procs = {'lulesh' : 8, 'cg' : 16, 'jacobi' : 20}
skip = {'lulesh' : 75, 'cg' : 15, 'jacobi' : 15}

for j in ['cg','lulesh', 'jacobi']: 
    for i in ['std','ms','fs','global']:
        path = 'data/'+j+'/stats-'+i+'.csv'
        my_data = pd.read_csv(path,comment='#')
        print(" == Sums for entry "+path+" ==")
        print(my_data['Duration'].sum(axis=0))
        print(my_data['Joules'].sum(axis=0)/procs.get(j))

