########################################################################################################################
# This code illustrates how to use the function "following"

# Files you need:
# (1) Following_Index.py
# (2) sample.csv

# Outputs:
# (1) heatmap: asymmetrical following indexes of industries

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities 1, 2340001 (2023).

########################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Following_Index import following, coverage

############################################## parameters ##############################################################
dmin = 200                                         # minimum percolation distance
dmax = 2000                                        # maximum percolation distance for the first average
delta = 200                                        # interval for the percolation distance
m = 1.0 + (dmax - dmin) / float(delta)
a = np.arange(dmin,dmax,delta)                     # a range for the percolation distance

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
FI = np.zeros((len(Cate),len(Cate)), dtype=float)
Cov = np.zeros(len(Cate), dtype=float)

for d in a:
    findex, catf, n_catf = following(xc,yc,catec,d)
    FI += findex / (len(a)+0.0)
    cover, catc, n_catc = coverage(xc,yc,catec,d)
    Cov += cover / (len(a)+0.0)

order = np.zeros(len(catf),dtype=int)
for c in range(len(catf)):
    order[c] = [i for i in range(len(catc)) if catc[i]==catf[c]][0]

Cov = Cov[order]
ca = np.where(countsCate >= 3)[0]                              # filtered and order the categories according to their coverage
coverage_filtered = Cov[ca]
index2 = np.argsort(coverage_filtered)                         # arrange the categories from low to high coverage
Co_sort2 = np.zeros((len(ca),len(ca)), dtype=float)
for row in range(len(ca)):
    Co_sort2[row,:] = FI[ca[index2[row]],ca[index2]]

fig1 = plt.figure(num=1, figsize=(9, 7.5), dpi=100, facecolor='w', edgecolor='k')
ax1 = fig1.add_subplot(1,1,1)
im1 = ax1.imshow(Co_sort2, cmap='RdYlGn', vmin=np.min(Co_sort2), vmax=np.max(Co_sort2))
cbar = plt.colorbar(im1)
plt.xticks(np.arange(0, len(ca)))
ax1.set_xticklabels(catf[ca[index2]], fontsize=11)
plt.yticks(np.arange(0, len(ca)))
ax1.set_yticklabels(catf[ca[index2]], fontsize=11)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('Following index', rotation=90, fontsize=13)
cbar.ax.get_yaxis().labelpad = 15
plt.xlabel('Category', fontsize=13)
plt.ylabel('Category', fontsize=13)
plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.99)
plt.savefig('FollowingIndex.png', format='png', dpi=300)
plt.show()
