########################################################################################################################
# This code computes the co-occurrence between pairs of industries. Here, a grid size is used for the partition of data.
# One can run the code with different gird size and take the average co-occurrence.

# Inputs:
# (1) xc: x-coordinates of the firms
# (2) yc: y-coordinates of the firms
# (3) catec: types of the firms (e.g., SSIC or category)
# (4) gridsize: width of cells in the grid

# Outputs:
# (1) cooc: pairwise co-occurrence of industries, a m x m matrix, with m being the number of categories or industries,
#           the (i,j) entry gives co-occurrence between industry cat[i] and industry cat[j]
# (2) cat: the industries
# (3) n_cat: number of firms in each industry

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities 1, 2340001 (2023).

# How to use the function: see Sample_Cooccurrence.py
########################################################################################################################

import numpy as np
import math


def cooccurrence(xc,yc,catec,gridsize):
    cat = np.unique(catec)
    n_cat = np.zeros(len(cat),dtype=int)                                         # number of company in a category
    xmax = np.max(xc)
    xmin = np.min(xc)
    ymax = np.max(yc)
    ymin = np.min(yc)
    M = int(math.ceil((xmax-xmin)/gridsize))+1
    N = int(math.ceil((ymax-ymin)/gridsize))+1
    perturbation = np.round(0.05*gridsize*np.random.random(1))
    x0 = xmin - perturbation                                                     # x-coordinate where the grid starts
    y0 = ymin - perturbation
    prob = np.zeros((len(cat),M*N),dtype=float)
    i = 0
    for k in cat:
        category_members = np.where(catec == k)[0]
        n_cat[i] = len(category_members)
        posx = xc[category_members]                                              # x-coordinates for the members
        posy = yc[category_members]
        grid = np.zeros((M,N), dtype=float)
        for coy in range(n_cat[i]):
            ix = math.floor((posx[coy]-x0)/float(gridsize))                      # check the grid the firm belong to
            iy = math.floor((posy[coy]-y0)/float(gridsize))
            grid[int(ix),int(iy)] += 1.0/float(n_cat[i])                         # frequency in each grid
        prob[i,:] = grid.ravel()                                                 # represent the probability with one-dimensional vector
        i += 1
    cooc = np.corrcoef(prob)                                                     # pair-wise correlation
    return cooc, cat, n_cat
