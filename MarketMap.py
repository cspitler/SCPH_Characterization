#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:44:00 2018

@author: riggs

Mostly derived from 
https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import matplotlib as mpl
import os
import pandas as pd

def BinaryMarketMap(priceDrop):
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    
    # draw state boundaries.
    # data from U.S Census Bureau
    shp_info = m.readshapefile(os.path.join('Market_Resources','st99_d00'),'states',drawbounds=True)
    
    # choose a color for each state based on population density.
    statenames=[]
    colors = []
    cmap = plt.cm.RdYlGn #colormap options https://matplotlib.org/examples/color/colormaps_reference.html
    vmin = -0.7
    vmax = 0.7 # set range.
    ATOLL_CUTOFF = 0.005
    ax = plt.gca()
    for i, shapedict in enumerate(m.states_info):
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            # Offset Alaska and Hawaii to the lower-left corner. 
            seg = m.states[int(shapedict['SHAPENUM'] - 1)]
            if statename == 'Alaska':
            # Alaska is too big. Scale it down to 35% first, then transate it. 
                seg = list(map(lambda x: (0.35*x[0] + 800000, 0.35*x[1]-1300000), seg))
            if statename == 'Hawaii' and float(shapedict['AREA']) > ATOLL_CUTOFF:
                seg = list(map(lambda x: (x[0] + 5100000, x[1]-1400000), seg))
                
            value = priceDrop.loc[statename][0]
            # calling colormap with value between 0 and 1 returns
            # rgba value. Take sqrt root to spread out colors more.
            noramlized = (value-vmin)/(vmax-vmin)
            color = rgb2hex(cmap(noramlized)[:3]) 
            colors.append(color)
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)
        statenames.append(statename)
    title = priceDrop.columns.tolist()[0]
    plt.savefig(os.path.join('Market_Resources',' '.join((title,'Map.png'))),
                dpi = 300, bbox_inches='tight')
    plt.close()
    
    #plt.title('Filling State Polygons by Population Density')
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(4, 1))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Price Reduction vs Fuel')
    plt.savefig(os.path.join('Market_Resources',' '.join((title,'Scale.png'))),
                dpi = 300, bbox_inches='tight')
    plt.close()
    
df = pd.read_csv(os.path.join('Market_Resources','Market Prices.csv'), index_col = 0)
#print(df[['Hybrid Com NG Price Change']])
BinaryMarketMap(df[['Hybrid Com NG Price Change']])
BinaryMarketMap(df[['Hybrid Ind NG Price Change']])

