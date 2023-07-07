########################################################################################################################
# This code illustrates how to use the function "agglomeration"

# Files you need:
# (1) Clustering_Agglomeration.py
# (2) sample.csv

# Outputs:
# (1) figure: average agglomeration of industries

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities 1, 2340001 (2023).

########################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Clustering_Agglomeration import agglomeration, agglomeration_grid

############################################## parameters ##############################################################
dmin = 200                                         # minimum percolation distance
dmax1 = 2000                                       # maximum percolation distance for the first average
dmax2 = 5000                                       # maximum percolation distance for the second average
delta = 200                                        # interval for the percolation distance
m = 1.0 + (dmax1 - dmin) / float(delta)
a = np.arange(dmin,dmax2,delta)                    # a range for the percolation distance

######################################### load data ####################################################################
cols = ['x coordinate', 'y coordinate',	'category']
dtypes = {'x coordinate': float, 'y coordinate': float,	'category': int}
data = pd.read_csv('sample.csv', usecols=cols, dtype=dtypes, skiprows=0)
xc = data['x coordinate']
yc = data['y coordinate']
catec = data['category']

xc = xc.to_numpy()
yc = yc.to_numpy()
catec = catec.to_numpy()

Cate, countsCate = np.unique(catec, return_counts=True)
Agg = np.zeros(len(Cate), dtype=float)                           # average agglomeration from 200 to 2000
Agg2 = np.zeros(len(Cate), dtype=float)                          # average agglomeration from 200 to 5000
Agg_grid = np.zeros(len(Cate), dtype=float)
for d in a:
    agg, cat, n_cat = agglomeration(xc,yc,catec,d)
    Agg2 += agg / (len(a)+0.0)
    if d <= dmax1:
        Agg += agg/m

fig1 = plt.figure(num=1, figsize=(6,4.5))
ax = fig1.add_subplot(111)
order = np.argsort(Agg)
plt.xlabel('Category', fontsize=10)
plt.ylabel('Average agglomeration', fontsize=10)
plt.plot(np.arange(0,len(Agg)), Agg[order], color='C1', marker='s', markersize=6, linewidth=0, fillstyle='none', markeredgewidth=1.5, label='Maximum percolation distance: ' + str(dmax1) + 'm')
plt.plot(np.arange(0,len(Agg2)), Agg2[order], color='C4', marker='1', markersize=6, linewidth=0, fillstyle='none', markeredgewidth=1.5, label='Maximum percolation distance: ' + str(dmax2) + 'm')
plt.legend(fontsize=10)
ax.set_xticks(np.arange(0,len(Agg)))
ax.set_xticklabels(cat[order], fontsize=9)
plt.ylim([0,1])
plt.xlim([-1,len(Agg)])
plt.subplots_adjust(top=0.94, bottom=0.12, left=0.14, right=0.94)
ax.tick_params(axis='both', which='major', labelsize=9)
plt.savefig('Agglomeration.png', format='png', dpi=300)

plt.show()
