import math
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress

class PVTsystem:
    def __init__(self, filename):
        raw = {}
        self.parameters = {}
        with open(filename) as f:
            for line in f.readlines():
                p,l,d,h = line.strip('\n').split(',')
                try:
                    raw[p] = {'low':float(l),
                                    'design':float(d),
                                    'high':float(h)}
                    self.parameters[p] = float(d)
                except:
                    raw[p] = {'low':l,'design':d,'high':h}
                    self.parameters[p] = d   
        self.AssignVariables()
        
    def AssignVariables(self):
        #raw inputs
        self.cspDegradation = self.parameters['CSP Degradation']
        self.pvDegradation = self.parameters['CPV Degradation']
        self.powerUse = self.parameters['Power use']
        self.collectorCost = self.parameters['Collector Cost']
        self.pvCost = self.parameters['PV Cost']
        self.markup = self.parameters['Indirect Markup']
        self.debtToEquity = self.parameters['Debt to equity']
        self.interestRate = self.parameters['Interest Rate']
        self.loanTerm = self.parameters['Loan term']
        self.taxRate = self.parameters['Effective Tax rate']
        self.discountRate= self.parameters['Nominal Discount Rate']
        self.pvOM = self.parameters['CPV O&M']
        self.cspOM= self.parameters['CSP O&M']
        self.omEscalation= self.parameters['O&M Escalation']
        self.DNI= self.parameters['Average DNI']
        self.lifetime = self.parameters['Years of operation']
        self.fuelCost= self.parameters['Price of fuel']
        self.elecCost= self.parameters['Cost of Electricity']
        self.MACRS= self.parameters['MACRS']
        self.fedITC= self.parameters['Federal ITC']
        self.stateITC= self.parameters['State ITC']
        self.statePBIduration = 0
        self.statePBI = 0 #monthly payments cent/kWh
        self.stateEPBB = 0 #lump sum upfront. Reduce cost of system [$/W]
        self.profit = self.parameters['Profit Margin']
        self.cspNameplate = self.parameters['Thermal Nameplate']
        self.elecFrac = self.parameters['Electric Fraction']
        self.heatFrac= self.parameters['Heat Fraction']
        
        self.calcLCOH(self.elecFrac,self.heatFrac)
        
    def calcLCOH(self, elecFraction = None, heatFraction = None, form = 'H'):
        #fractions should be in terms of power incident on the dish, not the receiver
        if elecFraction == None: elecFraction = self.elecFrac
        if heatFraction == None: heatFraction = self.heatFrac
        #calculates sizing of installation default 1MW
        collectorArea = self.cspNameplate/(heatFraction*0.001) #MW / 0.1MW per m2
        self.pvNameplate = collectorArea*elecFraction*0.001
        totalNameplate = self.pvNameplate+self.cspNameplate
        #calculates system cost based on type of system
        
        if form == 'H':                
            self.installCost = (self.pvCost+self.collectorCost)*(1+self.markup)*collectorArea
        elif form == 'PV' or heatFraction == 0:
            self.installCost = (self.pvCost)*(1+self.markup)*collectorArea
        elif form == 'TH' or elecFraction ==0:
            self.installCost = (self.collectorCost)*(1+self.markup)*collectorArea
        else:
            print('Invalid entries. Check calcLCOH call')
            quit()
       
    
        #depreciation schedule
        depSch = []
        if self.MACRS == 'no':
            depSch = [0.050, 0.0950,0.0855,0.0750,0.0693,0.0623,0.059,0.0590,
                       0.0591,0.0590,0.0591,0.0590,0.0591,0.0590,0.0591,0.0315]
            for i in range(int(self.lifetime)-len(depSch)):
                depSch.append(0)
        else:
            depSch = [0.20,0.32,0.192,0.1152,0.0576]
            for i in range(int(self.lifetime)-len(depSch)):
                depSch.append(0)    
        
        #loan payment setup
        Loan_Payment = -np.pmt(self.interestRate,
                               self.loanTerm,
                               self.installCost*self.debtToEquity)
              
        #Set NPV Lists
        PV = []
        CPV_gen = []
        CSP_gen = []
        PV_CSP = []
        PV_CPV = []
        OMCSP = []
        OMCPV = []
        OMHybrid = []
        Dep = []
        Principal = []
        Interest = []
        Balance = self.debtToEquity*self.installCost
        CPV_savings = []
        Subsidies = []
        Dep_shield = []
        Int_shield = []
        Tax_shield = []
        Cash_effect = []
        
        #calculates annual generation
        pvGen = self.DNI*365*elecFraction*collectorArea
        cspGen = self.DNI*365*heatFraction*collectorArea
        
        #first year subsidies calculated outside the loop
        Fed_sub = (self.installCost*self.fedITC)
        State_sub = (self.installCost*self.stateITC)
        lumpIncentive = self.stateEPBB*totalNameplate*1000000
        Subsidies.append(Fed_sub+State_sub+lumpIncentive)
        #Calculates lifetime financials for given parameters
        for i in range(int(self.lifetime)):
            CPV_gen.append((pvGen*((1-self.pvDegradation)**i)))
            CSP_gen.append((cspGen*((1-self.cspDegradation)**i)))
            PV_CPV.append((CPV_gen[i])/((1+self.discountRate)**(i+1)))
            PV_CSP.append((CSP_gen[i])/((1+self.discountRate)**(i+1)))
    
            
            OMCSP.append(self.cspOM*self.cspNameplate*1000*(1+self.omEscalation)**i)
            OMCPV.append(self.pvOM*self.pvNameplate*1000*(1+self.omEscalation)**i)
            OMHybrid.append(OMCSP[i] + OMCPV[i])
        
            Interest.append((Balance*self.interestRate))
            Principal.append((Loan_Payment-Interest[i]))
            Balance -= Principal[i]
            
            if pvGen==0 or cspGen==0: #if calculating pure LCOE or LCOH, ignore CPV savings
                CPV_savings.append(0)
            else:
                CPV_savings.append((CPV_gen[i]*self.elecCost))
               
            if i < self.statePBIduration:
                montlyIncentive = self.statePBI*(CSP_gen[i]+CPV_gen[i])
            else:
                montlyIncentive = 0
                
            if len(Subsidies)<self.lifetime: 
                Subsidies.append(montlyIncentive)
             
            Dep.append((depSch[i]*self.installCost))
            Dep_shield.append((Dep[i]*self.taxRate))
            Int_shield.append((Interest[i]*self.taxRate))
            Tax_shield.append(Dep_shield[i]+Int_shield[i])
        
            Cash_effect.append(((CPV_savings[i]+Tax_shield[i]+Subsidies[i]-OMHybrid[i]*(1-self.taxRate)-Principal[i]-Interest[i])))
            PV.append(((Cash_effect[i])/math.pow(1+self.discountRate,i+1)))      

        NPV = sum(PV)-self.installCost
    
        if pvGen != 0 and cspGen != 0:
            NPV_Energy = sum(PV_CSP)
        else:
            NPV_Energy = sum(PV_CSP)+sum(PV_CPV)
        
        self.LCOEnergy = -100*NPV/NPV_Energy
        self.priceChange = (self.LCOEnergy - self.fuelCost)/self.fuelCost
        
        fuel_displaced = (self.fuelCost*np.mean(cspGen))/100
        avg_savings = -np.mean(OMHybrid)+np.mean(CPV_savings)+np.mean(fuel_displaced)
        self.payback = (self.installCost + Loan_Payment)/avg_savings
        
def MarketAnalysis(pvtsystem, locationFile):
    d_MMBTU_c_kwh = 0.341 #converts  1 $/MMBTU to 0.341 cent/kWh
    d_gal_c_kwh = 3.7 #converts  1 $/gal to 3.7 cent/kWh
    
    market = ['Com','Ind']
    systems = {'Hybrid':[pvtsystem.elecFrac,pvtsystem.heatFrac],
               'CSP':[0,0.7]}
    
    marketDict = {}
    def stateLCOH(x):
        out = []
        label = []
        pvtsystem.DNI = x['DNI']
        if x['State']== 'California': 
            pvtsystem.stateEPBB = 0.6
        else:
            pvtsystem.stateEPBB = 0.0

        for s in systems.keys():
            for m in market:
                pvtsystem.elecCost = x[' '.join((m,'E c/kWh'))]/100
                pvtsystem.fuelCost = x[' '.join((m,'G $/MBTU'))]*d_MMBTU_c_kwh
                pvtsystem.calcLCOH(systems[s][0],systems[s][1],'H')
                out.extend([pvtsystem.LCOEnergy,pvtsystem.priceChange,pvtsystem.payback])
                label.extend([' '.join((s,m,'LCOH')),
                              ' '.join((s,m,'NG Price Change')),
                              ' '.join((s,m,'NG Payback [yrs]'))])
                pvtsystem.fuelCost = x[' '.join((m,'P $/Gal'))]*d_gal_c_kwh
                pvtsystem.calcLCOH(systems[s][0],systems[s][1],'H')
                out.extend([pvtsystem.priceChange,pvtsystem.payback])
                label.extend([' '.join((s,m,'Pro Price Change')),
                              ' '.join((s,m,'Pro Payback [yrs]'))])
        outDict = {}
        for k,v in zip(label,out):
            outDict[k]=v
        marketDict[x['State']]=outDict
        
    df = pd.read_csv(locationFile)
    df.apply(stateLCOH, axis = 1)
    marketDF = pd.DataFrame.from_dict(marketDict, orient='index')
    marketDF.to_csv(os.path.join('Market_Resources','Market Prices.csv'))
    #print(marketDF[['Hybrid Com NG Price Change']])
    
def classSensitivity(system, func, var, exclude = [], vary = 0.05, resolution = 10):
    sensitivityDict = {}
    for attr in dir(system):
        if not callable(getattr(system, attr)) and not attr.startswith("__"):
            if type(getattr(system,attr)) in [float,int] and attr not in exclude:
                base = getattr(system,attr)
                attrChange = {}
                for v in np.linspace(-vary,vary,resolution):
                    setattr(system,attr,base*(1+v))
                    getattr(system, func)()
                    attrChange[v] = getattr(system, var)
                sensitivityDict[attr]=attrChange
                setattr(system,attr,base)
    
    return(pd.DataFrame.from_dict(sensitivityDict))

def hitTarget(sensitivityDF, system = None, target = None):
    if target == None: target = sensitivityDF.mean()
    targetDF = systemSense[systemSense < target].dropna(axis = 1, how = 'all')
    if system:
        changeDict = {}
        for col in targetDF.columns:
            #find minimum index absolute value (smallest change)
            changes = targetDF[col].dropna().index.tolist()
            if all(c < 0 for c in changes):
                minChange = max(changes)
            elif all(c > 0 for c in changes):
                minChange = min(changes)
            else:
                changes = np.asarray(changes)
                minChange = changes[(np.abs(changes - 0)).argmin()] 
            
            changeDict[col] = {'base': getattr(system, col),
                              'Changed':getattr(system, col)*(1+minChange),
                              'Metric': targetDF.at[minChange,col]}
            
        return(pd.DataFrame.from_dict(changeDict, orient = 'index'))                        
    else:
        return(targetDF)
    
def improvementMap(sensitivityDF, system, var, func = None,  target = None, 
                   incr = 0.01, maxIter = 400, fixedCost = 1.5):
    if target == None: target = sensitivityDF.mean()
    trends = {}
    for col in sensitivityDF.columns:
        m, y, r, p, err = linregress(sensitivityDF.index, sensitivityDF[col])
        if r**2 > 0.95:
            trends[col] = {'type': 'linear', 'r^2':r**2, 'slope':m,'y':y,
                          'abs(slope)': abs(m),'cost':1.0, 'changes':0.0}
    trendDF = pd.DataFrame.from_dict(trends, orient = 'index')
    toTarget = getattr(system,var)-target
    
    original = (getattr(system,var))

    i = 0
    while toTarget>0:
        trendDF['$/chn'] = trendDF['cost']/trendDF['abs(slope)']
        trendDF.sort_values(['$/chn'], ascending = True, inplace = True)
        toTweak = trendDF.iloc[0]
        if trendDF.at[toTweak.name,'slope'] > 0:
            chg = -incr
        else:
            chg = incr
            
        trendDF.at[toTweak.name,'changes'] = trendDF.at[toTweak.name,'changes'] + chg
        trendDF.at[toTweak.name,'cost'] = trendDF.at[toTweak.name,'cost']*fixedCost
        #print(toTweak.name, trendDF.at[toTweak.name,'abs(slope)'])
        #print(chg)
        toTarget += chg*trendDF.at[toTweak.name,'abs(slope)']
        #print(toTarget)

        ''' 
        #Redo sensitivity analysis with new value
        #Seems unnecessary as the slopes did not significantly
        #Increases run time. Applicable for non-linear relations?
        
        base = getattr(system,toTweak.name)
        setattr(system,toTweak.name,base*(1+chg))
        
        exclude = ['elecCost','fedITC','interestRate','discountRate','taxRate']
        sensitivityDF = classSensitivity(system, 'calcLCOH', var = 'priceChange',exclude = exclude, 
                               vary = .9, resolution = 40)  
        for col in sensitivityDF.columns:
            m, y, r, p, err = linregress(systemSense.index, systemSense[col])
            if r**2 > 0.95:
                trendDF.at[col,'r^2'] = r**2
                trendDF.at[col,'slope'] = m
                trendDF.at[col,'y'] = y
                trendDF.at[col,'abs(slope)']=abs(m)    
        '''
        i+=1
        if i == maxIter:
            break
    
    for idx in trendDF.index:
        base = getattr(system,idx)
        change = trendDF.at[idx,'changes']
        setattr(system,idx, base*(1+change))
    
    system.calcLCOH()
    if toTarget > 0:
        print('Did Not Converge after {0:d} iterations'.format(maxIter))
        print('Start: ', original)
        print('Target: ', target)
        print('Final: ', getattr(system,var))
    else:
        print('Converged after {0:d} iterations'.format(i))
        return(trendDF.sort_values('changes',ascending = False))
    
system = PVTsystem(os.path.join('Market_Resources','DefaultTEA.csv'))

system.heatFrac = 0.42
system.elecFrac = .03
system.stateEPBB = 0.00
system.calcLCOH()

exclude = ['elecCost','fedITC','interestRate','discountRate','taxRate']
systemSense = classSensitivity(system, 'calcLCOH', var = 'priceChange',exclude = exclude, 
                               vary = .9, resolution = 40)
#print(hitTarget(systemSense, system, -0.0))
print(improvementMap(systemSense, system, var = 'priceChange', target = -0.6))

#members = [attr for attr in dir(system) if not callable(getattr(system, attr)) and not attr.startswith("__")]
#print(members)

#print(system.LCOEnergy)
#system.stateEPBB= 0.6
#system.statePBI = 0.3
#system.calcLCOH(system.elecFrac,system.heatFrac)
#print(system.LCOEnergy)
#MarketAnalysis(system,os.path.join('Market_Resources','State data.csv'))