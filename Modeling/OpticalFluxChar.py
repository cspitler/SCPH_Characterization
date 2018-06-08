# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 06:36:14 2018

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
import pandas as pd
import os

def export_map(array,filename,suffix,rng,palette,title = ''):
    plt.clf()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111,aspect='equal')
    ax1 = sns.heatmap(array,vmin = rng[0],vmax = rng[1],xticklabels=False, yticklabels=False,cmap=palette)
    ax1.set_title(title)
    print('Saving map to', os.getcwd() , ' '.join((filename,suffix)))
    fig1.savefig(' '.join((filename,suffix)), facecolor = 'white',dpi=90, bbox_inches='tight')  
    plt.close()
    return()

def Fluxmap(fluxFile,r_aperture,map_DNI,cells_per_side, artFile=None,
                offset_y = 0, offset_x = 0, exclude = []):

    N_channels = cells_per_side
    w_cell = 0.0055 #5.5mm in m
    delta_cells = 0.001 #1mm

    #fluxFile = os.path.join(('Flux_Maps',fluxFile))    
    #checks what type of flux map we're getting <Mod7 or after
    filePath = os.path.join('Flux_Maps',fluxFile)
    if fluxFile.lower().endswith('.xlsx'):
        print('Flux map for Mod<=6')
        xls = pd.ExcelFile(filePath)
        raw = xls.parse(header = None, index_col = None).as_matrix()
        head = raw[0,1:]
        pix_size = (float(max(head))-float(min(head)))/len(head)/1000
        raw = raw[1:,1:] #formatted for header on first row, first column
    elif fluxFile.lower().endswith('.csv'):
        print('Flux map for Mod>6')
        raw = np.genfromtxt(filePath,dtype=float, delimiter = ',')
        raw = raw[:len(raw)-2,1:] #formatted for headers on last row, first column        
        pix_size = 6*0.0254/len(raw) #6inch plate converted to mm divided by rows in map

    filename = '.'.join(fluxFile.split('.')[:-1])
    print('Read in flux file')

    raw = np.flipud(raw)
    #truncates plot to window area only
    length = len(raw)
    width = len(raw[0])
    w_map = int((2*r_aperture)/pix_size)
    r_map = int(w_map/2)
    
    center_y = int(length/2)+offset_y
    center_x = int(width/2)+offset_x #in pixels
    
    #map should be centered after naman's code. This is for manual adjustment
    #if statements make sure array doesn't run off the edge
 
    topY = center_y-r_map
    bottomY = center_y+r_map
    leftX = center_x-r_map
    rightX = center_x+r_map
    
    if topY <0: topY=0
    if leftX <0: leftX = 0
    if bottomY >= len(raw): bottomY =len(raw)-1
    if rightX >= len(raw[0]): rightX =len(raw[0])-1
    
    #raw = raw[topY:bottomY,leftX:rightX]
    window_area = raw[topY:bottomY,leftX:rightX]
    print('Trimmed to Window Area')
    #Makes arrays for mapping    
    abs_map = np.zeros_like(raw)
    refl_map = np.zeros_like(raw)
    tran_map = np.zeros_like(raw)
    window = np.zeros_like(raw)
    channels = np.zeros_like(raw)
    conc = np.zeros([cells_per_side,cells_per_side])
    power = np.zeros([cells_per_side,cells_per_side])
    
    if artFile:
        artPath = os.path.join('Module_Properties',artFile)

        art = np.genfromtxt('.'.join((artPath,'csv')),delimiter = ',',skip_header=1)
        artSave = artFile.split('.')[0]
        a_cell = art[0][1]
        r_cell = art[1][1]
        t_cell = art[2][1]
    
        a_bypass = art[0][2]
        r_bypass = art[1][2]
        t_bypass = art[2][2]
    
        a_channel = art[0][3]
        r_channel = art[1][3]
        t_channel = art[2][3]
        
    #Mark where the window is
    for i in range(len(window)):
        #y = round(r_map+(r_map**2-(i-r_map)**2)**(0.5))
        if r_map**2-(i-center_y)**2 >=0:
            x = round(center_x+(r_map**2-(i-center_y)**2)**(0.5))
            lx = 2*int(center_x)-x
            window[i,lx:x] = raw[i,lx:x]
            if artFile:
                abs_map[i,lx:x]=a_bypass
                refl_map[i,lx:x]=r_bypass
                tran_map[i,lx:x]=t_bypass
    
    print('Window Marked')
    
    #mark where channels are
    w_Channels = round((N_channels*w_cell+(N_channels-1)*delta_cells)/pix_size)
    start = m.floor(center_x-w_Channels/2)
    for i in range(len(window)):
        for j in range(N_channels):
            x1 = int(start+j*(w_cell+delta_cells)/pix_size)
            x2 = int(start+w_cell/pix_size+j*(w_cell+delta_cells)/pix_size)
            if artFile:
                abs_map[i,x1:x2]=[a_channel if x !=0 else x for x in abs_map[i,x1:x2]]
                refl_map[i,x1:x2]=[r_channel if x !=0 else x for x in refl_map[i,x1:x2]]
                tran_map[i,x1:x2]=[t_channel if x !=0 else x for x in tran_map[i,x1:x2]]  
            channels[i,x1:x2]=raw[i,x1:x2]
     
    print('Channels Marked')
    #Draws in cells
    w_array = round((N_channels*w_cell+(N_channels-1)*delta_cells)/pix_size)
    startx = int(center_x-w_array/2)
    starty = int(center_y-w_array/2)
    
    letterToNum = dict([(let,num) for num, let in enumerate('abcdefghijklmnopqrstuvwxyz',1)])
    rows = [letterToNum[x] for x in exclude if type(x) == str]
    
    columns = [x for x in exclude if type(x) == int]
    
    errorlist = []
    for i in range(cells_per_side):
        if i+1 not in rows:
            y1 = int(starty+i*(w_cell+delta_cells)/pix_size)
            y2 = int(starty+w_cell/pix_size+i*(w_cell+delta_cells)/pix_size)
        for j in range(cells_per_side):
            if j+1 not in columns:
                x1 = int(startx+j*(w_cell+delta_cells)/pix_size)
                x2 = int(startx+w_cell/pix_size+j*(w_cell+delta_cells)/pix_size)
            
            z = abs_map[y1:y2,x1:x2]
            #Known issue where np.any(z!=0) casts to shapes larger than they exist
            #print(y1,y2,x1,x2)
            fine = False
            aGood = False
            rGood = False
            tGood = False

            while not fine:
                try:
                    if artFile:
                        abs_map[y1:y2,x1:x2]=[a_cell if np.any(z !=0) else x for x in abs_map[y1:y2,x1:x2]]
                        aGood = True
                        refl_map[y1:y2,x1:x2]=[r_cell if np.any(z !=0) else x for x in refl_map[y1:y2,x1:x2]]
                        rGood = True
                        tran_map[y1:y2,x1:x2]=[t_cell if np.any(z !=0) else x for x in tran_map[y1:y2,x1:x2]]
                        tGood = True
                    fine = True
                except:
                    errorlist.append([i,j,y2-y1,x2-x1, aGood, rGood, tGood])
                    if y2-y1 > x2-x1: 
                        while y2-y1 > x2-x1: 
                            x2+=1
                    if y2-y1 < x2-x1: 
                        while y2-y1 < x2-x1: 
                            y2+=1
            rawxy = raw[y1:y2,x1:x2]            
            conc[i][j]=np.mean(rawxy/map_DNI)
            power[i][j]=np.sum(rawxy)*pix_size**2
    print('Cells Marked')
    print('Average cell concentration: ', np.mean(conc))
    print()
    
    print('Errors in cell assignement')
    errorDF = pd.DataFrame(errorlist, columns = ['Row','Column','dY','dX','a','r','t'])
    print(errorDF)    
    print()
    if artFile:
        
        absorbed = raw*abs_map
        reflected = raw*refl_map
        transmitted = raw*tran_map
        
        abs_power = np.sum(absorbed)*pix_size**2
        refl_power = np.sum(reflected)*pix_size**2
        tran_power = np.sum(transmitted)*pix_size**2
        
        artDF = pd.DataFrame()
        
        acellpow = (np.sum(absorbed[absorbed == raw*a_cell])*pix_size**2)
        rcellpow = (np.sum(reflected[reflected == raw*r_cell])*pix_size**2)
        tcellpow = (np.sum(transmitted[transmitted == raw*t_cell])*pix_size**2)

        achanpow =(np.sum(absorbed[absorbed == raw*a_channel])*pix_size**2)
        rchanpow = (np.sum(reflected[reflected == raw*r_channel])*pix_size**2)
        tchanpow = (np.sum(transmitted[transmitted == raw*t_channel])*pix_size**2)

        abypow = (np.sum(absorbed[absorbed == raw*a_bypass])*pix_size**2)
        rbypow = (np.sum(reflected[reflected == raw*r_bypass])*pix_size**2)
        tbypow = (np.sum(transmitted[transmitted == raw*t_bypass])*pix_size**2)
    
        artDF['Bypass No Chan'] = pd.Series([sum([abypow,rbypow,tbypow]),abypow,rbypow,tbypow])
        artDF['Bypass Chan'] = pd.Series([sum([achanpow,rchanpow,tchanpow]),achanpow,rchanpow,tchanpow])
        artDF['Cells'] = pd.Series([sum([acellpow,rcellpow,tcellpow]),acellpow,rcellpow,tcellpow])
        artDF['Total'] = artDF.apply((lambda x:sum(x)),axis = 1)
        artDF.set_index([['Power','Absorbed','Reflected','Transmitted']],inplace = True)
        artDF = artDF/artDF.get_value('Power','Total')
        print(artDF)
    
    total_power = np.sum(raw)*pix_size**2
    window_power = np.sum(window)*pix_size**2 
    cell_power = np.sum(power)    
    
    print('Cell Frac Calculation Check')
    print(abs_power+refl_power+tran_power, window_power)
    print(acellpow+rcellpow+tcellpow, cell_power)
    print(total_power)
    
    cell_frac = cell_power/window_power #for power split modeling
    spillage_frac = 1-window_power/total_power #for model comparison
    window_frac = window_power/total_power #to confirm calc and scale input power
        
    if not os.path.isdir(os.path.join('Optical_results',filename)):
        os.makedirs(os.path.join('Optical_results',filename))
    os.chdir(os.path.join('Optical_results',filename))

    np.savetxt(' '.join((filename,"conc.csv")),conc,delimiter=',')

    rng = [0,np.max(raw)]
    title = 'Total Map Power {0:.2f}W'.format(total_power)
    export_map(raw,filename,'raw map.png',rng,'nipy_spectral',title)

    title = 'Window Power {0:.2f}W, Cell Power {1:.2f}W, Cell fraction {2:.2f}'.format(window_power,cell_power,cell_frac)
    export_map(window,filename,'window map.png',rng,'nipy_spectral',title)
    
    window_area = window_area/map_DNI
    rng = [0,np.max(window_area)]
    title = 'Window Map'
    export_map(window_area,filename,'window scaled.png',rng,'nipy_spectral',title)
    
    title = 'Absorption Map'
    rng = [0,1]
    export_map(abs_map, filename, 'absorption map.png',rng,'Greys', title)
    
    title = 'Reflection Map'
    rng = [0,1]
    export_map(refl_map, filename, 'reflection map.png',rng,'Greys', title)
    
    title = 'Transmission Map'
    rng = [0,1]
    export_map(tran_map, filename, 'transmission map.png',rng,'Greys', title)
    
    to_return = {'spillage': spillage_frac,
                  'windowFraction': window_frac,
                  'bypass': (1-cell_frac),
                  'cellFraction':cell_frac}
    pd.DataFrame.from_dict(to_return, orient = 'index').to_csv(' '.join((filename,"stats.csv")))
                  
    to_return['cellConc'] = conc
    if artFile:
        artDF.to_csv(''.join((artSave,'.csv')))
        to_return['art'] = artDF

    os.chdir('..')
    os.chdir('..')

    return()
   
#exclusion list should have numbers for columns, letters in lower case for rows
#with starting index at 1 and 'a' respectively
Fluxmap('65.0_515DNI.csv', 0.15/2,515,11, 'Mod8_ART', exclude = [6])
#Fluxmap('58.5_844DNI.xlsx', 0.08/2,844,7,'Mod6_ART', -100,-100, exclude = 4)