# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:40:51 2017

@author: Riggs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import integrate
from scipy import stats
import string
import sys
from shutil import copyfile

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
        
    return(probes)
        
'''TEST PARAMETERS FOR CALCULATION'''

#Test parameters
masking = 0.35 
shading = 0.08 #on dish from receiver
mirror_loss = 0.11

test_date = '1-23'
fluxmap = '58.5_844DNI'
Mod = 6

filename = test_date

#Calibration constants
PV_flow_cal = (20/3-2/6)/(5)
TR_flow_cal = 200/(5*60)
PV_pres_cal = 15/0.12
GHI_cal = 5000
DNI_cal = 0.00000776
Cp_water = 4.184

TRwindow = 0.57 #shading after CPV before receiver, estimated
TReff = 0.95

CPV_mismatch = 0.085
CPV_wire_R = 0.126
Cell_area = 0.0055**2

#Imports guide to channel names
xls = pd.ExcelFile(os.path.join('Test_Data',test_date,'DATALOGGER HEADER.xlsx'))
channel_df = xls.parse()

#Imports metrics from IV sweeps
IV_df = pd.read_table(os.path.join('Test_Data',test_date,''.join((test_date,' Test 1.txt'))))
IV_df['time (min)'] = IV_df['time (s)    '].astype(float)/60 #
IV_inteveral = IV_df.at[1,'time (s)    ']-IV_df.at[0,'time (s)    ']
IV_df.set_index('time (min)',inplace=True)
IV_df.columns = ['Time s','Pmp','Imp','Vmp','Jsc','Isc','Voc','FF']
mask = IV_df.isin([np.inf,-np.inf])   
IV_df = IV_df.where(~mask, other=np.nan)

#Imports IV sweeps
sweep_df = pd.read_table(os.path.join('Test_Data',test_date,''.join((test_date,' Test 1-1.txt'))))
sweep_df['Voltage (V)'] = sweep_df['Voltage (V)'].str.replace(', ','')
sweep_df['Voltage (V)'] = sweep_df['Voltage (V)'].str.replace('[','')
sweep_df['Voltage (V)'] = sweep_df['Voltage (V)'].str.replace(']','')
starts = sweep_df.loc[sweep_df['Voltage (V)'].astype(float)%100 == 0]
starts = starts[starts['Voltage (V)'].astype(float)>0]
sweep_idx = starts.index.values.tolist()

#Imports mpp log
mpp_df = pd.read_table(os.path.join('Test_Data',test_date,''.join((test_date,' Test 1-1-1.txt'))))
mpp_df['Elapsed Min']=mpp_df['Time (s)']/60
mpp_df.set_index('Elapsed Min',inplace=True)

#Imports data logger data
df = pd.read_csv(os.path.join('Test_Data',test_date,''.join((test_date,' Test 1.csv'))),sep = '\t')

#Renames channels with descriptor names
assign = dict(zip(channel_df['CHANNEL'],channel_df['ASSIGNMENT']))
df.rename(columns = assign,inplace=True)

#Converts time into elapsed time in Minutes
df['Date'], df['Time'] = df['Time'].str.split(' ', 1).str
df['Hr'], df['Min'], df['Sec'],df['msec']  =  df['Time'].str.split(':').str
df['Sec'] = df['Sec'].astype(int)+df['msec'].astype(int)/1000
df['Min']= df['Min'].astype(int)+df['Sec'].astype(float)/60
df['Hr']= df['Hr'].astype(int)+df['Min'].astype(float)/60
df['Elapsed Min'] = (df['Hr'].astype(float)-min(df['Hr']))*60
df.set_index('Elapsed Min',inplace=True)


print('Module Parameters from Model')
source_loc = os.path.join('Test_Data',test_date,'sources')
sfiles = []

if not os.path.isdir(source_loc):
    os.makedirs(os.path.join(source_loc))
    
mod_stat_file = os.path.join('Modeling','Module_Properties',' '.join((fluxmap,str(Mod),'stats.csv')))
copyfile(mod_stat_file,os.path.join(source_loc,'Mod Stats.csv'))
sfiles.append(mod_stat_file)
modStatsDF = pd.read_csv(mod_stat_file, index_col = 0)
print(modStatsDF)
modStats = modStatsDF.to_dict(orient = 'index')
for k,v in modStats.items():
    modStats[k]= v['0']

CPV_heat = float(modStats['heatFraction'])
CPV_elec = float(modStats['elecFraction'])

TCs ={}
for TC in modStats['TCS'].strip('}{').split(','):
    pos,temp = TC.split(':')#[0].strip(" '")
    TCs[pos.strip(" '")] = temp.strip()

TC_pos = list(TCs.keys())
TCs = pd.DataFrame.from_dict(TCs, orient = 'index')
del modStats['TCS']
print()

print('Concentration by TC probes')
opt_file = os.path.join('Modeling','Optical_results',fluxmap,' '.join((fluxmap,'conc.csv')))
array_conc = np.genfromtxt(opt_file, delimiter = ',')
copyfile(opt_file,os.path.join(source_loc,'Concentration Stats.csv'))
sfiles.append(opt_file)

probes = pd.DataFrame.from_dict(dict(zip(TC_pos,TC_probe(TC_pos,array_conc))), orient = 'index')
probes['TC Names'] = ['PV TC1','PV TC2','PV TC3','PV TC4','PV TC5']
probes.rename({0:'conc'},axis = 1, inplace = True)
probes = probes.join(TCs)
probes.rename({0:'temp'}, axis = 1, inplace = True)
probes.reset_index(inplace = True)
probes.rename({'index':'Loc'}, axis = 1, inplace = True)
probes.set_index('TC Names',  inplace = True )
probes = probes.append(pd.DataFrame([['None',probes['conc'].mean(),probes['temp'].astype(float).mean()]],
                         columns = ['Loc','conc','temp'],index = ['Avg']))
print(probes)
print()

print('Stats on Flux Map')
flux_file = os.path.join('Modeling','Optical_results',fluxmap,' '.join((fluxmap,'stats.csv')))
fluxStats = pd.read_csv(flux_file, header= None, skiprows = 1, index_col = 0)
copyfile(flux_file,os.path.join(source_loc,'Flux Map Stats.csv'))
sfiles.append(flux_file)
print(fluxStats)
print()
fluxStats = fluxStats.to_dict(orient = 'index')
for k,v in fluxStats.items():
    fluxStats[k]= v[1]
window_frac = fluxStats['windowFraction']
cell_frac = fluxStats['cellFraction']

print('Module Normalized Power ART')
modFile = os.path.join('Modeling','Optical_results',fluxmap,''.join(('Mod',str(Mod),'_ART.csv')))
copyfile(modFile,os.path.join(source_loc,'ART Stats.csv'))
sfiles.append(modFile)
modART = pd.read_csv(modFile, index_col = 0)
print(modART)
print()
CPV_refl = modART.at['Reflected','Total']
CPV_tran = modART.at['Transmitted','Total']

Source_files = open(os.path.join(source_loc,'Source files.txt'),'w')

for item in sfiles:
    Source_files.write("{:s}\n".format(item))
Source_files.close()

if not os.path.isdir(os.path.join('Test_Data',test_date,'results')):
    os.makedirs(os.path.join('Test_Data',test_date,'results'))
os.chdir(os.path.join('Test_Data',test_date,'results'))

'''CALCULATIONS AND PLOTTING SECTION'''

df['DNI'] = df['DNI Sensor (V)'].astype(float)/DNI_cal
df['Pin'] = df['DNI'].astype(float)*masking*(1-shading)*(1-mirror_loss)*1.65**2

data_df = df.copy(deep=True)
data_df['Spillage Adj P'] = df['Pin']*TRwindow
data_df['TR Flow']=data_df['TR Flow V']*TR_flow_cal
data_df['PV Flow']=data_df['PV Flow V']*PV_flow_cal
data_df['PV Pres inlet'] = data_df['PV Inlet P Sensor (V)']*PV_pres_cal
data_df['PV Pres outlet'] = data_df['PV Outlet P Sensor (V)']*PV_pres_cal
data_df['PV dPres']= data_df['PV Pres outlet']-data_df['PV Pres inlet']
todrop = ['Date','Hr','Min','Sec','msec','TR Flow V','PV Flow V',
          'PV Outlet P Sensor (V)','PV Inlet P Sensor (V)']
data_df = data_df.drop(todrop, 1)
data_df.to_excel('_'.join((filename,'calculations.xlsx')))

#Power plot
powerflow_df = pd.DataFrame(columns = ['Receiver','PV Cooling'])
TR_flow = df['TR Flow V'].mean()*TR_flow_cal
powerflow_df['Receiver']=(df['TR Outlet'].astype(float)-df['TR Inlet'].astype(float))*TR_flow*Cp_water
powerflow_df['PV Cooling'] = (df['PV Outlet'].astype(float)-df['PV Inlet'].astype(float))*df['PV Flow V']*Cp_water
powerflow_df = pd.concat([powerflow_df,mpp_df['Power (W)'].astype(float)],axis =1)
powerflow_df.rename(columns = {'Power (W)':'PV Power'},inplace=True)
powerflow_df['R Loss'] = (CPV_wire_R*mpp_df['Current (A)']**2)
powerflow_df['Mis Loss'] = data_df['Spillage Adj P']*cell_frac*CPV_elec*CPV_mismatch

#remove negative and interpolates missing data
sm = 5
powerflow_df = powerflow_df[powerflow_df>0]
powerflow_df['Reflection'] = (df['Pin']*window_frac*CPV_refl)
powerflow_df['Reflection'] = powerflow_df['Reflection'].interpolate()
powerflow_df['PV Power'] = powerflow_df['PV Power'].interpolate()
powerflow_df['Transmitted'] = (powerflow_df['Receiver']/TReff).interpolate()
powerflow_df['PV Cooling'] = powerflow_df['PV Cooling'].interpolate()
powerflow_df['R Loss'] = powerflow_df['R Loss'].interpolate()
powerflow_df['Mis Loss'] = powerflow_df['Mis Loss'].interpolate()
powerflow_df['Elec Loss'] = powerflow_df[['R Loss','Mis Loss']].sum(axis = 1)

powerflow_df = powerflow_df[powerflow_df>0]
powerplot_df=powerflow_df[['PV Power','PV Cooling','Transmitted','Reflection']]

print('Missing Power')
data_df['Error'] = (data_df['Spillage Adj P']-powerplot_df.sum(axis = 1))/data_df['Spillage Adj P']
print(data_df['Error'].median())
print()

#Modeled Power flow
modeled = pd.DataFrame()
modeled['PV Power'] = pd.Series(CPV_elec*cell_frac*data_df['Spillage Adj P'].mean()*(1-CPV_mismatch))
modeled['PV Cooling'] = pd.Series(CPV_heat*cell_frac*data_df['Spillage Adj P'].mean())
modeled['Transmitted'] = pd.Series(CPV_tran*data_df['Spillage Adj P'].mean())
modeled['Reflection'] = pd.Series(df['Pin'].mean()*window_frac*CPV_refl)
modeled['Elec Loss'] = powerflow_df['Elec Loss'].mean()
modeled.to_excel('Modeled power plot.xlsx')

error_dict={}
for c in powerplot_df.columns.tolist():
    exp =powerplot_df[c].mean()
    mod = modeled[c].mean()
    error_dict[c]={'Exp':exp,'Mod':mod,'frac delta' : (exp-mod)/mod}
    
error_df = pd.DataFrame.from_dict(error_dict, orient = 'index')
print(error_df)
print()

##Plot power
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw = {'width_ratios':[3, 1]})

powerplot_df.plot.area(ax= ax1, legend=False)
ax1.plot(df.index,data_df['Spillage Adj P'], label ='Net to Receiver', color = 'k')
ax1.set_title('Experimental')
ax1.set_xlabel('Elapsed Time / min')
ax1.set_ylabel('Cummulative Power / W')
ax1.set_ylim([0,max(data_df['Spillage Adj P'])*1.20])
ax1.set_xlim([0,450])

modeled.plot.bar(ax = ax2,stacked=True,legend = False, width = 30)
ax2.tick_params(
    axis='x',         # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
ax2.set_title('Modeled')

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 0.4))

plt.savefig('Power Plot.png',figsize = (8,8), facecolor = 'white',dpi=300, bbox_inches='tight')
plt.close()

##Fraction of total input
TRfrac = (powerflow_df['Transmitted']/data_df['Spillage Adj P']).median()
PVfrac = (powerflow_df['PV Power']/data_df['Spillage Adj P']).median()
Coolingfrac = (powerflow_df['PV Cooling']/data_df['Spillage Adj P']).median()
Reflection = (powerflow_df['Reflection']/data_df['Spillage Adj P']).median()
losses = 1-(TRfrac+PVfrac+Coolingfrac+Reflection)

print('Power Flow Synopsis')
print('TR {0:0.3f}, PVE {1:0.3f}, PVH {2:0.3f}, Refl {3:0.3f}, Loss {4:0.3f}'.format(TRfrac, PVfrac, Coolingfrac,Reflection, losses))
print()
powerflow_df.to_excel('_'.join((filename,'powerflow.xlsx')))

#Economic Analysis based on test data
Pin = integrate.trapz(data_df['Spillage Adj P'],data_df.index)/60/1000
nonspill = integrate.trapz(data_df['Spillage Adj P'],data_df.index)/60/1000

Heat = integrate.trapz(powerflow_df['Transmitted'][pd.notnull(powerflow_df['Transmitted'])],dx=0.25)/60/1000
Elec = integrate.trapz(powerflow_df['PV Power'][pd.notnull(powerflow_df['PV Power'])],dx=0.25)/60/1000
print('Total Heat [kWh]',Heat)
print('Total Electricity [kWh]',Elec)


#Create a specific temperature dataframe for analysis
temp_df = df[['DNI','PV TC1','PV TC2','PV TC3','PV TC5',
              'PV Inlet','PV Outlet']].copy(deep=True)
temp_df =temp_df[temp_df>0] #filters out any buggy data

temp_df.to_excel('_'.join((filename,'Module Temperatures.xlsx')))
TCs = ['PV TC1','PV TC2', 'PV TC3','PV TC5']
maxes = TCs + ['DNI','PV Outlet','PV Inlet']

for mx in maxes:
    print(mx, round(temp_df[mx].max(),2), round(temp_df[mx].idxmax(),2))
    
#Plot of temperaturesplt.title('DNI')
#plt.plot(temp_df.index,temp_df['DNI']/10, label = 'DNI/10')
plt.plot(temp_df.index,temp_df['PV TC1'], label = 'TC1')
plt.plot(temp_df.index,temp_df['PV TC2'], label = 'TC2')
#plt.plot(temp_df.index,temp_df['PV TC3'])
plt.plot(temp_df.index,temp_df['PV TC5'], label = 'TC3')
plt.plot(temp_df.index,temp_df['PV Inlet'])
plt.plot(temp_df.index,temp_df['PV Outlet'])

plt.title('Cell Temperatures')
plt.ylabel('Temperature / C')
plt.xlabel('Elapsed Time / min')
plt.ylim([0,max(df['DNI']/10)*1.1])
plt.legend(loc =0)
plt.savefig('PV cell temperatures.png',facecolor = 'white',dpi=300, bbox_inches='tight')
plt.close()

'''
#TR and PV inlet outlet
plt.title('PV Cooling and TR Flow Temperatures')
plt.plot(df.index,df['PV Outlet'], label = 'PV Outlet')
plt.plot(df.index,df['PV Inlet'], label = 'PV Inlet')
plt.plot(df.index,df['TR Outlet'], label = 'TR Outlet')
plt.plot(df.index,df['TR Inlet'], label = 'TR Inlet')
plt.legend()
plt.ylabel('Temperature / deg C')
plt.xlabel('Elapsed Time / min')
plt.ylim([0,max(df['TR Outlet'])*1.5])
plt.savefig('PV cooling and TR flow.png',facecolor = 'white',dpi=90, bbox_inches='tight')
plt.close()
'''

'''
#Plots IV curves
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in range(len(sweep_idx)):
    if i %5==0:
        try:
            tdf = sweep_df[sweep_idx[i]:sweep_idx[i+1]]
            tdf = tdf.drop([sweep_idx[i], sweep_idx[i]+1])
            timestamp = (i+1)*IV_inteveral/60
            ax1.scatter(tdf['Voltage (V)'].astype(float),tdf['Current (A)'].astype(float),label = '{0:0.2f}'.format(timestamp))
        except:
            pass
        
plt.title('IV curves')
plt.xlabel('Voltage / V')
plt.ylabel('Current / A')
start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(0, end, 1))
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.legend(title = 'Time/min',loc='lower left', fontsize = 12)
plt.savefig('IV Curves.png',facecolor = 'white',dpi=90, bbox_inches='tight')
plt.close()
'''

'''
#DNI and GHI
plt.title('DNI')
plt.plot(df.index,df['DNI'])
plt.plot(df.index,df['GHI1']*GHI_cal)
plt.plot(df.index,df['GHI2']*GHI_cal)
plt.plot(df.index,df['GHI3']*GHI_cal)
plt.plot(df.index,df['GHI4']*GHI_cal)
plt.title('Solar Resource')
plt.ylabel('Irradiance / W/m$^2$')
plt.xlabel('Elapsed Time / min')
plt.ylim([0,max(df['DNI'])*1.1])
plt.legend(loc ='upper right')
plt.savefig('Solar Resource.png',facecolor = 'white',dpi=90, bbox_inches='tight')
plt.close()
'''

#DNI vs Cell Temp
start = 0
maxDNI = temp_df['DNI'].idxmax()
temp_df = temp_df[temp_df['DNI']>500]
temp_df['Avg']=temp_df[['PV TC1','PV TC2','PV TC5']].mean(axis=1)
#temp_df = temp_df[temp_df.index >60]
morning = temp_df.loc[:maxDNI,:]
#afternoon = temp_df.loc[maxDNI:,:]

Avg_df=pd.DataFrame()
Avg_df['DNI'] = temp_df['DNI']
Avg_df['Avg dT'] = temp_df['Avg']-temp_df['PV Inlet']
Avg_df.to_excel('TvDNI plot.xlsx')
TCs = ['Avg','PV TC1','PV TC2', 'PV TC5']

print(probes)
print()
ks = pd.DataFrame(columns = ['TC','conc','Morning','err','Afternoon','err'])
for tc in TCs:
    
    plt.scatter(morning['DNI'],morning[tc]-morning['PV Inlet'],alpha = 0.35,label='morning', color = 'blue')
    slope, intercept, mr_value, p_value, std_err = stats.linregress(morning['DNI'],morning[tc]-morning['PV Inlet'])
    mk =slope/(CPV_heat*probes.at[tc,'conc']*Cell_area)
    merr = std_err/(CPV_heat*probes.at[tc,'conc']*Cell_area)
    plt.plot(df['DNI'],intercept+slope*df['DNI'], label ='morning fit', color = 'g')
    
    maxTC = temp_df[tc].idxmax()
    afternoon = temp_df.loc[maxDNI:,:]

    plt.scatter(afternoon['DNI'],afternoon[tc]-afternoon['PV Inlet'],alpha = 0.35,label='afternoon', color = 'm')
    slope, intercept, ar_value, p_value, std_err = stats.linregress(afternoon['DNI'],afternoon[tc]-afternoon['PV Inlet'])
    ak  = slope/(probes.at[tc,'conc']*Cell_area*CPV_heat)
    aerr = std_err/(CPV_heat*probes.at[tc,'conc']*Cell_area)

    plt.plot(df['DNI'],intercept+slope*df['DNI'], label = 'afternoon fit', color = 'red')

    k = pd.DataFrame([[tc, probes.at[tc,'conc'], mk,mr_value**2,ak,ar_value**2]],
                     columns = ['TC','conc','Morning','err','Afternoon','err'])
    ks = ks.append(k)
    
    plt.title('{0:s} Cell Temp Correlation'.format(tc))
    plt.legend(loc='upper left')
    plt.ylabel('$delta$ Temperature / deg C')
    plt.xlabel('DNI / W/m$^2$')
    plt.ylim([0,max(temp_df[tc]-temp_df['PV Inlet'])*1.1])
    plt.xlim([500,max(temp_df['DNI'])*1.1]) #sets x-scale, currently manual for appropriate range
    plt.savefig('{0:s} cell temp correlation.png'.format(tc),
                facecolor = 'white',dpi=90, bbox_inches='tight')
    plt.close()
ks.set_index('TC',inplace=True)

print(ks)
print()
#print(ks[1:]['Afternoon'].describe())

#DNI vs PV stats
JvDNI_df = pd.DataFrame()
JvDNI_df['DNI']=df['DNI']
JvDNI_df = JvDNI_df.combine_first(IV_df[['Jsc','Voc','FF','Pmp']])
JvDNI_df['DNI']=JvDNI_df['DNI'].interpolate()
JvDNI_df['Full_Spectrum']=powerflow_df['PV Power']/(data_df['Spillage Adj P']*cell_frac)
JvDNI_df['Full_Spectrum'] = JvDNI_df['Full_Spectrum'].interpolate()
JvDNI_df['In_band']=JvDNI_df['Full_Spectrum']/0.638
JvDNI_df = JvDNI_df[JvDNI_df.index >2]
JvDNI_df.dropna(axis=0,inplace=True) 

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.5, hspace=None)

JvDNI_df.plot.scatter('DNI','Jsc',ax= ax1, color = 'k',legend=False)
ax1.set_ylabel('J$_{sc}$ [mA/cm$^2$]')
ax1.set_ylim([0,max(JvDNI_df['Jsc'])*1.1])

JvDNI_df.plot.scatter('DNI','Voc',ax= ax2, color = 'green', legend=False)
ax2.set_ylabel('V$_{oc}$ [V]')
ax2.set_ylim([0,max(JvDNI_df['Voc'])*1.1])

JvDNI_df.plot.scatter('DNI','FF',ax= ax3, color = 'm',legend=False)
ax3.set_ylabel('FF [%]')
ax3.set_ylim([0,max(JvDNI_df['FF'])*1.1])

JvDNI_df.plot.scatter('DNI','Full_Spectrum',ax= ax4, color = 'red',label = 'Full Spectrum')
JvDNI_df.plot.scatter('DNI','In_band',ax= ax4,label = 'In-band')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels,loc = 'best', fontsize = 8)
ax4.set_ylabel('Eff [%]')
ax4.set_ylim([0,max(JvDNI_df['In_band'])*1.1])

plt.savefig('DNI correlations.png',facecolor = 'white',dpi=200)
plt.close()

#Time vs PV stats
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.5, hspace=None)

JvDNI_df.reset_index(inplace=True)
JvDNI_df.rename({'index':'Time [min]'},axis = 1,inplace=True)

JvDNI_df.plot.scatter('Time [min]','Jsc',ax= ax1,color = 'k', legend=False)
ax1.set_ylabel('J$_{sc}$ [mA/cm$^2$]')
ax1.set_ylim([0,max(JvDNI_df['Jsc'])*1.1])

JvDNI_df.plot.scatter('Time [min]','Voc',ax= ax2, color = 'green', legend=False)
ax2.set_ylabel('V$_{oc}$ [V]')
ax2.set_ylim([0,max(JvDNI_df['Voc'])*1.1])

JvDNI_df.plot.scatter('Time [min]','FF',ax= ax3,color = 'm', legend=False)
ax3.set_ylabel('FF [%]')
ax3.set_ylim([0,max(JvDNI_df['FF'])*1.1])

JvDNI_df.plot.scatter('Time [min]','Full_Spectrum',ax= ax4, color = 'red', label = 'Full Spectrum')
JvDNI_df.plot.scatter('Time [min]','In_band',ax= ax4,label = 'In-band')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels,loc = 'best', fontsize = 8)
ax4.set_ylabel('Eff [%]')
ax4.set_ylim([0,max(JvDNI_df['In_band'])*1.1])

plt.savefig('PV Stats over time.png',facecolor = 'white',dpi=200, bbox_inches='tight')
plt.close()
