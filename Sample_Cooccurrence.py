########################################################################################################################
# This code illustrates how to use the function "cooccurrence"

# Files you need:
# (1) Cooccurrence.py
# (2) sample.csv

# Outputs:
# (1) heatmap: paiwise cooccurrence of industries

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities 1, 2340001 (2023).

########################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Cooccurrence import cooccurrence
import seaborn

############################################## parameters ##############################################################
dmin = 200                                         # minimum grid size
dmax = 5000                                        # maximum grid size
delta = 200                                        # interval
a = np.arange(dmin,dmax+1,delta)                   # a range for the grid size

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
Co_occurrence = np.zeros((len(Cate),len(Cate)), dtype=float)
for gridsize in a:
    cooc, cat, n_cat = cooccurrence(xc,yc,catec,gridsize)
    Co_occurrence += cooc / (len(a)+0.0)

ca = np.where(n_cat >= 3)[0]                                         # include only category with 3 data points and more
Co_filtered = np.zeros((len(ca),len(ca)), dtype=float)
for row in range(len(ca)):
    Co_filtered[row,:] = Co_occurrence[ca[row],ca]
cg_filtered = seaborn.clustermap(Co_filtered, method='single', metric='cosine', cmap='RdYlGn', xticklabels=ca, yticklabels=ca)
order_filtered = cg_filtered.dendrogram_row.reordered_ind

Co_fsort = np.zeros((len(ca),len(ca)), dtype=float)
for row in range(len(ca)):
    Co_fsort[row,:] = Co_filtered[order_filtered[row],order_filtered]

fig3 = plt.figure(num=3, figsize=(9, 7.5), dpi=100, facecolor='w', edgecolor='k')
ax3 = fig3.add_subplot(1,1,1)
im3 = ax3.imshow(Co_fsort, cmap='RdYlGn', vmin=np.min(Co_filtered), vmax=np.max(Co_filtered))
cbar = plt.colorbar(im3)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('Cooccurence', rotation=90, fontsize=13)
cbar.ax.get_yaxis().labelpad = 15
plt.xticks(np.arange(0, len(ca)))
ax3.set_xticklabels(cat[ca[order_filtered]], fontsize=11)
plt.yticks(np.arange(0, len(ca)))
ax3.set_yticklabels(cat[ca[order_filtered]], fontsize=11)
plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.99)
plt.xlabel('Category', fontsize=13)
plt.ylabel('Category', fontsize=13)
plt.savefig('Cooccurrence.png', format='png', dpi=300)

plt.show()
