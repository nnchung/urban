########################################################################################################################
# This code computes the following indexes between pairs of industries. Here, a distance is used to define whether a
# firm follows another in close proximity. One can run the code with different distances and take the average.

# Inputs:
# (1) xc: x-coordinates of the firms
# (2) yc: y-coordinates of the firms
# (3) catec: types of the firms (e.g., SSIC or category)
# (4) d: distance

# Outputs:
# (1) FI: asymmetrical following indexes of industries, a m x m matrix, with m being the number of categories or
#         industries, the (i,j) entry quantify the degree industry cat[i] follows industry cat[j]
# (2) cat: the industries
# (3) n_cat: number of firms in each industry
# (4) cover_scaled: coverage of the industries

# Reference: Lock Yue Chew, Ning Ning Chung, Wen Xuan Sia, Hoai Nguyen Huynh, Glenn Sim, Alvin Chua, and Zhongwen Huang.
# A Data Analytic Elucidation on the Spatial Agglomeration of Singapore Maritime Industry.
# International Journal on Smart and Sustainable Cities, 2340001 (2023).

# How to use the function: see Sample_Following_Index.py
########################################################################################################################

import numpy as np
import math


def following(xc,yc,catec,d):
    cat, n_cat = np.unique(catec, return_counts=True)
    DistanceMetric = np.zeros((len(xc),len(xc)), dtype=float)
    PairDistance = []
    for firm1 in range(len(xc)-1):
        for firm2 in np.arange(firm1+1,len(xc)):
            DistanceMetric[firm1,firm2] = np.sqrt((xc[firm1]-xc[firm2])**2 + (yc[firm1]-yc[firm2])**2)
            DistanceMetric[firm2,firm1] = np.sqrt((xc[firm1]-xc[firm2])**2 + (yc[firm1]-yc[firm2])**2)
            PairDistance.append(np.sqrt((xc[firm1]-xc[firm2])**2 + (yc[firm1]-yc[firm2])**2))

    FI = np.zeros((len(cat),len(cat)), dtype=float)
    for A in range(len(cat)):
        for B in range(len(cat)):
            catA = cat[A]
            catB = cat[B]
            membersA = np.where(catec == catA)[0]                    # members in category A
            membersB = np.where(catec == catB)[0]                    # members in category B
            COA = 0.0
            for memb in membersA:                                    # go through all members in category A
                if A == B:                                           # for same category
                    excl = np.setdiff1d(membersB,memb)               # exclude the member itself
                    D_AB = DistanceMetric[memb,excl]                 # get distance with other members
                else:
                    D_AB = DistanceMetric[memb,membersB]             # get distance between the member in category A and all members in category B
                nearby = np.where(D_AB <= d)[0]                      # find distance smaller than d
                if len(nearby) > 0:
                    COA += 1.0
            COA = COA/float(len(membersA))
            FI[A,B] = COA                                            # FI[A,B] quantify the degree A follows B
            COB = 0.0
            for memb in membersB:
                if A == B:
                    excl = np.setdiff1d(membersA,memb)
                    D_AB = DistanceMetric[memb,excl]
                else:
                    D_AB = DistanceMetric[memb,membersA]
                nearby = np.where(D_AB <= d)[0]
                if len(nearby) > 0:
                    COB += 1.0
            COB = COB/float(len(membersB))
            FI[B,A] = COB                                            # FI[B,A] quantify the degree B follows A
    return FI, cat, n_cat


def coverage(xc,yc,catec,d):
    cat, n_cat = np.unique(catec, return_counts=True)
    xmax = np.max(xc)
    xmin = np.min(xc)
    ymax = np.max(yc)
    ymin = np.min(yc)
    M = int(math.ceil((xmax-xmin)/d))+1
    N = int(math.ceil((ymax-ymin)/d))+1
    grid_all = np.zeros((M,N),dtype=float)
    cover = np.zeros(len(cat), dtype=float)
    cover_scaled = np.zeros(len(cat), dtype=float)
    for k in range(len(cat)):
        grid = np.zeros((M,N),dtype=float)                           # grid the space
        members = np.where(catec == cat[k])[0]
        for coy in members:
            ix = math.floor((xc[coy]-xmin)/float(d))                 # get the gird that the data point belong to
            iy = math.floor((yc[coy]-ymin)/float(d))
            grid[int(ix),int(iy)] += 1.0                             # add the data point to the grid
            grid_all[int(ix),int(iy)] += 1.0                         # this grid includes data in all categories
        cover[k] = np.count_nonzero(grid)/float(grid.size)           # count the number of non-empty grid, i.e. the number of grid that the category covers
        cover_scaled[k] = np.count_nonzero(grid)
    cover_scaled = cover_scaled/float(np.count_nonzero(grid_all))
    return cover_scaled, cat, n_cat


