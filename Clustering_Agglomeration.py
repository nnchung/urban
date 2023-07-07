########################################################################################################################
# This code computes agglomeration of each industry. Here, a percolation thresholds is used to for the clustering of
# data points (via DBSCAN). Agglomeration is calculated for one percolation distance. One can run the code with
# different percolation distances and take the average to yield an average agglomeration. Use agglomeration_grid when
# the data size is big.

# Inputs:
# (1) xc: x-coordinates of the firms
# (2) yc: y-coordinates of the firms
# (3) catec: types of the firms (e.g., SSIC or category)
# (4) d: percolation distance

# Outputs:
# (1) agg: agglomeration of all types of industries, agg[i] gives clustering agglomeration of cat[i]
# (2) cat: the industries
# (3) n_cat: number of firm in each industry

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities, 2340001 (2023).

# How to use the function: see Sample_Agglomeration.py
########################################################################################################################

from sklearn.cluster import DBSCAN
import numpy as np
import math


def agglomeration(xc,yc,catec,d):
    cat = np.unique(catec)
    n_cat = np.zeros(len(cat))                                            # number of company in a category
    agg = np.zeros(len(cat),dtype=float)
    i = 0
    for k in cat:                                                         # go through all categories
        category_members = np.where(catec == k)[0]                        # find members in category k
        n_cat[i] = len(category_members)                                  # number of firms in category k
        posx = xc[category_members]                                       # x-coordinates for the members
        posy = yc[category_members]
        XY = np.vstack((posx,posy))
        XY = XY.T
        db = DBSCAN(eps=d,min_samples=1).fit(XY)                          # clustering data points with dbscan
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_                                               # record the cluster that each data point belong to
        cluster, counts_cluster = np.unique(labels, return_counts=True)   # count the size of each cluster
        f2 = 0
        for j in counts_cluster:
            f2 += (j/float(np.sum(counts_cluster)))**2                    # sum of frequency square
        agg[i] = f2
        i += 1
    return agg, cat, n_cat


def agglomeration_grid(xc,yc,catec,d):
    cat, n_cat = np.unique(catec, return_counts=True)
    xmax = np.max(xc)
    xmin = np.min(xc)
    ymax = np.max(yc)
    ymin = np.min(yc)
    gridsize = 0.05*d                                                            # use a larger grid size (preferably <=0.1*d) for shorter computation time
    M1 = int(math.ceil((xmax-xmin)/gridsize))+1
    M2 = int(math.ceil((ymax-ymin)/gridsize))+1
    data_rep = []
    fraction = []
    data_len = []
    for k in range(len(cat)):
        category_members = np.where(catec == cat[k])[0]
        posx = xc[category_members]                                              # x-coordinates for the members
        posy = yc[category_members]
        grid = np.zeros((M1,M2), dtype=float)
        rep = []                                                                 # represent all data in a grid with 1 data point
        rep_grid = []
        for coy in range(int(n_cat[k])):
            ix = math.floor((posx[coy]-xmin)/float(gridsize))                    # check the grid the firm belong to
            iy = math.floor((posy[coy]-ymin)/float(gridsize))
            if grid[int(ix),int(iy)] == 0:
                rep.append(category_members[coy])
                rep_grid.append([ix,iy])
            grid[int(ix),int(iy)] += 1.0                                         # number of data in each grid
        data_rep.append(rep)
        fraction.append([grid[int(ix),int(iy)]/float(n_cat[k]) for [ix,iy] in rep_grid])
        data_len.append(len(rep))

    agg = np.zeros(len(cat))
    for k in range(len(cat)):                                                    # go through all categories
        category_members = data_rep[k]                                           # find members in category k
        fk = fraction[k]
        posx = xc[category_members]                                              # x-coordinates for the members
        posy = yc[category_members]
        XY = np.vstack((posx,posy))
        XY = XY.T
        db = DBSCAN(eps=d,min_samples=1).fit(XY)                                 # clustering data points with dbscan
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_                                                      # record the cluster that each data point belong to
        freq = []
        for cc in np.unique(labels):                                             # for each cluster, calculate the total frequency
            freq_c = [fk[ii] for ii in range(len(labels)) if labels[ii]==cc]
            freq.append(np.sum(freq_c))
        f2 = 0
        for j in freq:
            f2 += j**2                                                           # sum of frequency square
        agg[k] = f2
    return agg, cat, n_cat
