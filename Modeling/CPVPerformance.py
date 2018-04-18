# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 08:41:09 2017

@author: Khepri
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math as m
from shutil import copyfile
import string
import pandas as pd

# -*- coding: utf-8 -*-

def SetParameters(param):
    global T_amb
    global T_cell_lim
    global m_dot
    global t_saph
    global t_opad
    global k_saph
    global k_opad
    global k_air
    global bub_frac
    global p_conc
    global DNI
    global d_channel
    
    T_amb = float(param[0][2])+273
    T_cell_lim = float(param[1][2])+273
    m_dot = float(param[2][2])
    t_saph = float(param[3][2])
    t_opad = float(param[4][2])
    k_saph = float(param[5][2])
    k_opad = float(param[6][2])
    k_air = float(param[7][2])
    bub_frac = float(param[8][2])
    p_conc = float(param[9][2])
    DNI = float(param[10][2])
    d_channel = float(param[11][2])
    return()
    
def SolarSpectrum(eff,art):
    global heatfraction
    global elecfraction
    global eff_scaling

    
    inband_fraction = .638
    eff_scaling = 1
    energy = EnergyFlow(inband_fraction,eff,art)
    elecfraction=energy[0]
    heatfraction=energy[1]
    return()

def XenonSpectrum(eff,art):
    global heatfraction
    global elecfraction
    global eff_scaling
    
    inband_fraction = 0.7992
    eff_scaling = 0.926
    energy = EnergyFlow(inband_fraction,eff,art)
    elecfraction=energy[0]
    heatfraction=energy[1]
    return()

def EnergyFlow(inband_fraction,Eff,ART):
    
    Eff = Eff*eff_scaling
    inband_Eff = Eff/inband_fraction
        
    reflect = ART[6][8]
    
    elec = inband_Eff*inband_fraction*ART[10][4]
   
    inband_heat = (1-inband_Eff)*inband_fraction*ART[10][4]
    outband_heat = (1-inband_fraction)*ART[11][4]
    sub_heat = ART[12][1]+ART[12][2]+ART[12][3]+ART[12][5]+ART[12][6]+ART[12][7]+ART[12][8]
    
    trans = ART[3][8]
    
    heat = inband_heat+outband_heat+sub_heat

    total = reflect+elec+heat+trans
    
    print('ART Energy Balance check  ')
    
    print('Inband eff: ', np.mean(inband_Eff))
    print('Full Spectrum eff: ',np.mean(Eff))        
    print('Reflected light: ',np.mean(reflect))
    print('Electricity Fraction: ',np.mean(elec))
    print('Heat Fractoin: ',np.mean(heat))
    print('Transmitted light: ',np.mean(trans))
    print('Total: ',np.mean(total))
    print()
    
    
    return(elec,heat)    

def bubbles(low, high,bound,bubblefile=''):
    if bubblefile:
        bub_dist = np.genfromtxt('.'.join((bubblefile,'csv')),dtype=float,delimiter =',')
        k_comp = 1/((bub_dist/k_air)+((1-bub_dist)/k_opad))
    else:
        area = np.random.uniform(low,high,size=(7,7))
        k_comp = np.empty_like(area)
        if bound == 'low':
            k_comp = 1/((area/k_air)+((1-area)/k_opad))
        if bound =='high':
            k_comp = area*k_air+(1-area)*k_opad
            
    return(k_comp)

def thermalcalc(report=''):
    global T_w_outlet_avg
    global r_aperture
    
    A_c = w_channel*d_channel # [m2] x-sectional area PER channel
    D_H = 4*A_c/(2*w_channel + 2*d_channel) # [m] hydraulic diameter of channel
   
    l_HT = N_channels*w_cell + (N_channels - 1)*delta_cells # [m] length of heated part of channel, for temperature plotting
    l_channel = r_aperture # [m] length of channel (not just of HT, used for pressure drop)
    SA_HT = w_cell**2 # [m^2] heat transfer area based on area of CPV cells because
    # thickness of interface is small compared to gaps between cells
    Nu_D = 5.39 # From Bergman & Lavine p. 553 Table 8.1 for parallel plates, isoflux, one insulated.

    T_W = np.empty_like(PVheat)
    V_W = np.empty_like(PVheat)
    Re_D_w = np.empty_like(PVheat)
    Pr = np.empty_like(PVheat)
    R_water = np.empty_like(PVheat)
    R_opad = np.empty_like(PVheat)
    R_bottom = np.empty_like(PVheat)
    T_cell = np.empty_like(PVheat)
    
    rho_w = 992.2 #kg/m3
    mu_w = 0.0006527 #Pa s
    cp_w = 4187 # J/kg/K
    k_w = 0.62856 # W/m/K
    
    k_quartz = 1.38 #W/mK
    
    for i in range(len(T_W)):
        for j in range(len(T_W[0])):
            if j == 0:
                T_W[i][j] = T_w_inlet+PVheat[i][j]/(cp_w*m_dot/N_channels) #dT in K
           
            else:
                T_W[i][j] = T_W[i][j-1]+PVheat[i][j]/(cp_w*m_dot/N_channels) #dT in K
    
            V_W[i][j] = m_dot/N_channels/rho_w/A_c
            Re_D_w[i][j] = rho_w*V_W[i][j]*D_H/mu_w # [] Reynolds number for internal flow
            Pr[i][j] = cp_w*mu_w/k_w # [] Prandtl number 
            
            h = Nu_D*k_w/D_H # [W/m^2/K]
            R_water[i][j] = 1/(h*SA_HT)
            R_glass = t_saph/k_saph/SA_HT # [K/W] thermal resistance of glass conduction
            R_opad[i][j] = t_opad/k_comp[i][j]/SA_HT
            R_bottom[i][j] = R_glass+R_opad[i][j]+R_water[i][j]
    
    R_opad_top = 0.000075/0.2/SA_HT
    R_quartz = 0.003/k_quartz/SA_HT
    
    R_top = R_opad_top +R_quartz
    
    #Convection off top Quartz
    kin_visc = 0.0000185 #kg/m*s
    H = 2*r_aperture
    therm_dif = 1.9*10**-5 #m^2/s
    k_air = 0.0261 #W/mC
    CTE_air = 0.0034 #1/K
    Win_area = m.pi*r_aperture**2

    Pr_air = kin_visc/therm_dif

    emm = 0.92
    sig = 5.6703*10**-8
    
    Ts = T_amb 
    for i in range(10):
        Gr = 9.8 *(CTE_air)*(Ts - T_amb)*H**3 / (kin_visc**2)
        Ra = (Gr * Pr_air)
        Nu_air = 0.68 + (0.670*Ra**(1/4))/(1+(0.492/Pr_air)**(9/16))**(4/9)
        h_air = Nu_air*k_air/(2*r_aperture)
        R_air = 1/(h_air*Win_area)
        R_rad = (emm*sig*Win_area*(Ts**2+T_amb**2)*(Ts+T_amb))**(-1)
        R_surface = (1/(1/R_air+1/R_rad))
        Ts = T_amb + np.mean(PVheat)*R_surface
    
    R_top = R_opad_top +R_quartz + R_surface
    T_cell_not = PVheat*R_top+T_amb    
    Ts = np.mean(T_cell_not)
        
    #Tf = (Tp + Ta) / 2
    #Rho = Rhoref (Tref +273) / (Tf + 273)  
    
    R_opad_side = 0.001/0.2/(0.0055*0.0005)
    R_side = R_opad_side

    R_total = 1/(1/R_top+1/R_bottom)
    
    #Other Calculations
    T_w_outlet_avg = np.average(T_W[:,-1])
    T_mean = np.average(T_W)    
    Q_dis = np.sum(PVheat)
    # The equation for thermal entry length in laminar flow is eq. 8.23 from
    # Bergman & Lavine, p. 524
    Re_Dwater = np.average(Re_D_w)
    Vel = m_dot/N_channels/rho_w/A_c # [m/s] flow velocity (same in all channels)

    Prandtl = np.average(Pr)
    #Vel = np.average(V_W)
    f_moody = 64/Re_Dwater # Moody friction factor
    dPdx = f_moody*rho_w*Vel**2/2/D_H # by definition of friction factor
    dP = (dPdx*l_channel/1000)*0.145038 #pressure drop in PSI
    P_pump = dPdx*l_channel*(m_dot/rho_w)
                             
    x_fdt = D_H*0.05*Re_Dwater*Prandtl # [m]
    # For hydrodynamic entry length, laminar flow, eq. 8.3, p. 519
    x_fdh = D_H*0.05*Re_Dwater # [m]
    
    '''Cell Temp Calculation'''
    T_drop = PVheat*R_total
    T_cell_cooled = T_drop+T_W
        
    Tmax = np.max(T_cell_not-273)
    Tmin = np.min([T_cell_cooled])
    print(Tmax-273, np.max(T_cell_cooled)-273)
    bins = np.linspace(Tmin,Tmax,20+1)
    colors = sns.color_palette("coolwarm",20).as_hex()
     
    heatmap(T_cell_cooled,bins,colors,'cooled')
    heatmap(T_cell_not,bins,colors,'not cooled')
            
    if report == 1:
        '''Calculation reporting'''
        print('''The thermal and hydrodynamic entry lenths are {0:.4f} and {1:.4f}, respectively'''.format(x_fdt,x_fdh))
        print('Reynolds number is {0:.1f}'.format(Re_Dwater))
        print('The pressure drop in the channels is {0:.2f} PSI'.format(dP))
        print('pumping power is {0:.3f} W'.format(P_pump))
        print('fluid velocity is {0:.2f} m/s'.format(Vel))
        print('water exit temp is {0:.0f} C'.format(T_w_outlet_avg-273))
        print('Hydraulic diameter is {0:.0f} um'.format(D_H*1000000))
        print('Water mean temp of {0:.0f} C, {1:.0f} K'.format(T_mean - 273, T_mean))
        print('length of channel is {0:.3f} m'.format(l_channel))
        print()
        print('Total Heat dissipated',Q_dis)
        print('Cell Heat range',np.min(PVheat),np.max(PVheat))
        print()
        print('R saph', np.mean(R_glass))
        print('R encap', np.mean(R_opad))
        print('R water', np.mean(R_water))
        print('R bottom', np.mean(R_total))
        print('R_top',R_top)
        print('R_side',R_side)
        print('Total Modeled thermal resistance:', np.average(R_total))
        print(TC_probe(TC_pos,R_total))
        print()
        
    calculated_props['topR']= R_top
    calculated_props['bottomR']= np.mean(R_total)
    calculated_props['sideR'] = R_side

    return(T_cell_cooled-273,dP)

'''Plots a heat map for the cell array'''

def heatmap(array,bins, colors, savefile):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    pad = 0.005
    ax1.set_xlim([0-pad, r_aperture*2+pad])
    ax1.set_ylim([0-pad, r_aperture*2+pad])
    
    array[3] = np.nan
    array[:,3] = np.nan

    tmax = int(np.nanmax(array))-273
    tmin = int(np.nanmin(array))-273
    tavg = int(np.nanmean(array))-273
    colormap = np.empty_like(array,dtype=object)

    for i in range(len(colormap)):
        for j in range(len(colormap[i])):
            for k in range(len(colors)):
                if array[i][j]>=bins[k] and array[i][j]<=bins[k+1]:
                    colormap[i][j] = colors[k]
                    
    window = plt.Circle((r_aperture,r_aperture), r_aperture, color='k', fill = False)
    ax1.add_artist(window)
    
    xstart = r_aperture-(w_cell/2)-(w_cell+delta_cells)*(3)
    ystart = r_aperture-(w_cell/2)+(w_cell+delta_cells)*(3)
    print(xstart,ystart)

    #textoffset = w_cell*0.2
    
    for i in range(cells_per_side):
        for j in range(cells_per_side):
            if i!=3 and j!=3:
                posx = xstart+j*(w_cell+delta_cells)
                posy = ystart-i*(w_cell+delta_cells)
                ax1.add_patch(
                patches.Rectangle(
                    (posx, posy),   # (x,y)
                    w_cell,          # width
                    w_cell,          # height
                    fill=True,
                    color = colormap[i][j])
                )
            #ax1.annotate('{0:.0f}'.format(array[i][j]),
            #             xy=(posx+textoffset,posy+textoffset))
    #bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="blue", ec="b", lw=2)
    #ax1.text(r_aperture, ystart+2*w_cell, "Coolant Flow", color='w',
    #         ha="center", va="center", rotation=0,size=10, bbox=bbox_props)
    #ax1.text(r_aperture-2*w_cell,ystart+2*w_cell,'{0:.0f} C'.format(T_w_inlet-273),ha="right",va="center")
    #ax1.text(r_aperture+2*w_cell,ystart+2*w_cell,'{0:.0f} C'.format(T_w_outlet_avg-273),ha="left",va="center")
    print('plot heat map')
    #print(win_power,rad_power,rad_power/win_power)
    #plt.title('Window Power:{0:.0f}W\nPV Input power: {1:.0f}W\nCell Frac.{2:.2f} '.format(win_power,rad_power,rad_power/win_power))
    plt.axis('off')
    deg= u'\N{DEGREE SIGN}'
    plt.title('Tmax: {0:d}\N{DEGREE SIGN}C,Tmin: {1:d}\N{DEGREE SIGN}C,Tavg: {2:d}\N{DEGREE SIGN}C'.format(tmax,tmin,tavg))
    fig1.savefig(' '.join((savefile,'temp map.jpg')), facecolor = 'white',dpi=200, bbox_inches='tight')
    plt.close()
    np.savetxt(' '.join((savefile,"temp map.csv")),array,delimiter=',')

    
    
def electrical(conc):
    global I
    global V
    global Eff
    
    T= 80
    area = 10000*w_cell**2

    I = (0.0085*conc+0.0102)*area
    V = 1*e(-10)*conc**3 - 4*e(-7)*conc**2 + 0.0002*conc + 3.3872
    Eff = 2*e(-11)*conc**3 - 5*e(-8)*conc**2 + 4*e(-5)*conc + 0.4786
    Irow = []
    Vrow = []
    Prow = []
    for i in range(len(V[0])):
        Vrow.append(min(V[:,i]))
        Irow.append(np.sum(I[:,i]))
        Prow.append(Irow[i]*Vrow[i])
        
    #I = np.vstack((I,Irow))
    #V = np.vstack((V,Vrow))
    
    np.savetxt(' '.join((scenario,"I map.csv")),I,delimiter=',')
    np.savetxt(' '.join((scenario,"V map.csv")),V,delimiter=',')
    np.savetxt(' '.join((scenario,"Eff map.csv")),Eff,delimiter=',')
    
    return()
    
def e(x):
    return(10**x)

def TC_probe(locations,array):

    TCs = np.empty((len(locations),4),dtype=str)
    
    d = dict(enumerate(string.ascii_uppercase, 1))
    d = {k: v for v, k in d.items()}
    
    for i in range(len(TCs)):
        TCs[i]=list(locations[i])
    
    for i in range(len(TCs)):
        for j in range(len(TCs[i])):
            try:
                TCs[i][j] = d[TCs[i][j]]
            except:
                pass
          
    TCs = TCs.astype(int) - 1

    probes = []
    for i in range(len(TCs)):
        Cell1 = (array[TCs[i][0],TCs[i][2]])
        Cell2 = (array[TCs[i][0],TCs[i][3]])
        Cell3 = (array[TCs[i][1],TCs[i][2]])
        Cell4 = (array[TCs[i][1],TCs[i][3]])
        probes.append(np.mean([Cell1,Cell2,Cell3,Cell4]))
    
    
    return(dict(zip(locations,probes)))

def MaxFlux():
    global p_conc
    global bub_frac
    global DNI
    
    T_lim = 105
    
    A_c = w_channel*d_channel # [m2] x-sectional area PER channel
    D_H = 4*A_c/(2*w_channel + 2*d_channel) # [m] hydraulic diameter of channel
    l_HT = N_channels*w_cell + (N_channels - 1)*delta_cells # [m] length of heated part of channel, for temperature plotting
    l_channel = r_aperture # [m] length of channel (not just of HT, used for pressure drop)
    SA_HT = w_cell**2 # [m^2] heat transfer area based on area of CPV cells because
    # thickness of interface is small compared to gaps between cells
    Nu_D = 5.39 # From Bergman & Lavine p. 553 Table 8.1 for parallel plates, isoflux, one insulated.
    
    rho_w = 992.2 #kg/m3
    mu_w = 0.0006527 #Pa s
    cp_w = 4187 # J/kg/K
    k_w = 0.62856 # W/m/K


    Q = p_conc*(w_cell**2)*DNI
    #rho_w = CP.PropsSI('D','T',T_w_inlet,'Q',0,'Water') # [kg/m3] denisty

    #cp_w = CP.PropsSI('CPMASS','T',T_w_inlet,'D',rho_w,'Water') # [J/kg/K] specific heat
    T_w = T_w_inlet + (Q/(cp_w*m_dot))
    #rho_w = CP.PropsSI('D','T',T_w,'Q',0,'Water') # [kg/m3] denisty
    #k_w = CP.PropsSI('L','T',T_w,'D',rho_w,'Water') # [W/m/K] thermal cond.
    h = Nu_D*k_w/D_H # [W/m^2/K]
    
    R_water = 1/(h*SA_HT)
    R_glass = t_saph/k_saph/SA_HT # [K/W] thermal resistance of glass conduction
    
    k_opad = np.mean(bubbles(bub_frac,bub_frac,'low'))
    R_opad = t_opad/k_opad/SA_HT
    R_total = R_glass+R_opad+R_water
    T_w-=273
    T_drop = T_lim - T_w

    Heat = T_drop/R_total #This is power of HEAT on cells. Not total flux on cells
    Pow = Heat/np.mean(heatfraction)
    flux = Pow/(w_cell**2)
    return(flux/1000)
    
############################################################################
'''Sets up file and report system'''

fluxFolder = '58.5_844DNI'
module = 6

module_file = 'Mod6_Design.csv'
Cell_art = 'Mod6_CellART.csv'
Cell_eff = .22

calculated_props = {}

'''Imports module parameters'''
param = np.genfromtxt((os.path.join('Module_Properties',module_file)),dtype='S20', delimiter = ',')
SetParameters(param)

#m_dot = 0.071 #flow rate of bigger pump

'''Set all system-level parameters:'''
T_w_inlet = T_amb # [K] inlet water temperature
T_w_inlet = 29+273 # [K] inlet water temperature

'''Set all geometry parameters:'''
r_aperture = 0.08/2
w_cell = 0.0055 # [m] width/length of cell
cells_per_side = 7
N_channels = cells_per_side # [] total number of cooling channels
delta_cells = 0.001 # [m] distance between cells
w_channel = w_cell # [m] set each channel width to cell width
k_comp = bubbles(0,0,'low')#,bub_dist)

TC_pos = ['AB22','BC66','DD44','EF22','FG55']

'''Sets spectrum to be used'''
ART = np.genfromtxt(os.path.join('Module_Properties',Cell_art),dtype = float, delimiter = ',')
SolarSpectrum(Cell_eff,ART)

#XenonSpectrum('.'.join((Cell_eff,'csv')),'.'.join((Cell_art,'csv')))

'''Calculates conc, power, and heat for each cell'''
concdist = np.genfromtxt(os.path.join('Optical_results',fluxFolder,' '.join((fluxFolder,'conc.csv'))),
                         dtype = float, delimiter = ',')
powerdist = concdist*DNI*(w_cell**2)
PVheat = powerdist*heatfraction
PVelec = powerdist*elecfraction

calculated_props['heatFraction'] = np.mean(heatfraction)
calculated_props['elecFraction'] = np.mean(elecfraction)

PVheat[3,:]=0
PVheat[:,3]=0

'''Thermal characterization calculations under each cell'''
Temp_map = thermalcalc(1)[0]
print(np.min(PVheat), np.max(PVheat))
print(np.mean(Temp_map))
print(np.mean(concdist))

'''Plots cell temperatures'''

Tmax = np.max([Temp_map])
Tmin = np.min([Temp_map])

bins = np.linspace(Tmin,Tmax,20+1)
colors = sns.color_palette("coolwarm",20).as_hex()
 
#heatmap(Temp_map,bins,colors,os.path('))


TCs = TC_probe(TC_pos,Temp_map)
calculated_props['TCS'] = TCs

savefile = os.path.join('Module_Properties',' '.join((fluxFolder,str(module),"stats.csv")))
pd.DataFrame.from_dict(calculated_props, orient = 'index').to_csv(savefile)
#np.savetxt(' '.join((scenario,"Temp map.csv")),Temp_map,delimiter=',')



