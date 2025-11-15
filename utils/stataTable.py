### Functions used in regression_projections.py
### Jessica Wan (j4wan@ucsd.edu)
### 12/2/21

# Import libraries
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MaxNLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.dates as mdates
from matplotlib import ticker
import re

## Function to read in STATA tables and parse coefficients, variables, and standard errors
class stataTable:
    def __init__(self,path,sheet,column,log_inc=False,min_sig=0):
        self.file_path = path
        self.sheet_name = sheet
        self.column_number = column
        self.min_sig = min_sig
               
        # rename consistently each type of coefficient labeling
        if log_inc == False:
            vmap = {'Income'            :'income',
                    'gridmean_income'   :'income',  
                    'Temperature'       :'T',    
                    'temperature'       :'T',
                    'gridmean_temp'     :'Tbar',    
                    'Precipitation'     :'P',
                    'precipitation'     :'P',
                    'gridmean_precip'   :'Pbar', 
                    'temp'              :'T',
                    'precip'            :'P', 
                    'Constant'          :'1'
                    }
            
            # rename consistently for square terms 
            vmapsq = {'Income Squared'        :'income*income',
                    'income_q2'             :'income*income',
                    'Temperature Squared'   :'T*T',    
                    'temp_q2'               :'T*T',
                    'Ttrend2'               :'Ttrend*Ttrend',    
                    'Tanom_q2'              :'Tanom*Tanom',    
                    'Precipitation Squared' :'P*P',
                    'precip_q2'             :'P*P',
                    'Ptrend2'               :'Ptrend*Ptrend',
                    'Panom_q2'              :'Panom*Panom',    
                    }
        # set up the naming map for the models ran with log income 
        elif log_inc == True:
            vmap = {#'Income'            :'income',
                    'gridmean_income'   :'np.log(income)',  
                    'Temperature'       :'T',    
                    'temperature'       :'T',
                    'gridmean_temp'     :'Tbar',    
                    'Precipitation'     :'P',
                    'precipitation'     :'P',
                    'gridmean_precip'   :'Pbar', 
                    'temp'              :'T',
                    'precip'            :'P', 
                    'Constant'          :'1'
                    }
            
            # rename consistently for square terms 
            vmapsq = {#'Income Squared'        :'income*income',
                    #'income_q2'             :'np.log(income)*np.log(income)',
                    'gridmean_income_q2'    :'np.log(income)*np.log(income)',  
                    'Temperature Squared'   :'T*T',    
                    'temp_q2'               :'T*T',
                    'Ttrend2'               :'Ttrend*Ttrend',    
                    'Tanom_q2'              :'Tanom*Tanom',    
                    'Precipitation Squared' :'P*P',
                    'precip_q2'             :'P*P',
                    'Ptrend2'               :'Ptrend*Ptrend',
                    'Panom_q2'              :'Panom*Panom',    
                    }            
        #   vmapsq = {k.lower():v for k,v in vmapsq.items()}
        #   vmap   = {k.lower():v for k,v in vmap.items()}

        self.vmapsq = vmapsq
        self.vmap = vmap

    def load(self,exclude_vars=[]):
        if self.sheet_name:
            full = pd.read_excel(self.file_path,sheet_name=self.sheet_name)
        else:
            full = pd.read_excel(self.file_path)
        variablerow = np.argwhere((full.iloc[:,0]=='VARIABLES').values)[0][0]
        colnumrow = full.iloc[variablerow-1,:].str.extract('(\d+)')
        colidx = np.where(colnumrow==str(self.column_number))[0][0]
        
        obsi = np.where(full.iloc[:,0]=='Observations')[0][0]

        codei = np.where(full.iloc[:,0].str.startswith('CODE UTILIZED')==True)

        terms = full.iloc[4:obsi,0].dropna()
        termcoefs = full.iloc[terms.index,colidx]

        usei = termcoefs.dropna().index
        useterms = full.loc[usei].iloc[:,0]
        usecoefs = full.loc[usei].iloc[:,colidx].str.replace('\*','', regex=True).astype(float)
        usecoefs_err = full.loc[usei+1].iloc[:,colidx].str.replace('\(|\)','', regex=True).astype(float)
        usecoefs_sig = full.loc[usei].iloc[:,colidx].str.count('\*')

        coefnum = pd.DataFrame(['c'+str(i) for i in range(len(useterms))],index=useterms.index)[0]

        coefdict_long = {useterms.loc[ind]:usecoefs.loc[ind] for ind in useterms.index}
        coefdict = {coefnum.loc[ind]:usecoefs.loc[ind] for ind in useterms.index}

        coefdict_long_err = {useterms.loc[ind]:usecoefs_err.loc[ind+1] for ind in useterms.index}
        coefdict_err = {coefnum.loc[ind]:usecoefs_err.loc[ind+1] for ind in useterms.index}

        coefdict_long_sig = {useterms.loc[ind]:usecoefs_sig.loc[ind] for ind in useterms.index}
        coefdict_sig = {coefnum.loc[ind]:usecoefs_sig.loc[ind] for ind in useterms.index}
        
        termnames = [] 
        # rename varnames to match the climate input data 
        for i,t in useterms.items():
            t = t.replace(' ##','*')
            t = t.replace('#','*')
            t = t.replace('Gridded ','')
            t = t.replace('c.','')
            t = t.replace('_mon_','')
            t = t.replace('_mon','')
            t = t.replace('precipitation_', 'precipitation')
            t = t.replace('temperature_',   'temperature')
            if t.endswith('_'):
                t=t[:-1]

            for vm in self.vmapsq:
                if vm.lower() in t.lower():
                    t = t.replace(vm,self.vmapsq[vm])
            for vm in self.vmap:
                if vm.lower() in t.lower():
                    t = t.replace(vm,self.vmap[vm])

            termnames.append(t)
                
        termnames = pd.Series(termnames,index=useterms.index)
        
        termnamesu = termnames.str.replace('\*|\(|\)|np\.log',' ', regex=True).str.split()
        #termnamesu = termnames.str.replace('\(|\)|np\.log',' ', regex=True).str.split('\*')
        termnamesu = np.unique(termnamesu.apply(pd.Series).stack().reset_index(drop=True))
        termnamesu = termnamesu.tolist()
        termnamesu.remove('1')
        
        include_i = [i for i,v in termnames.items() if v not in exclude_vars]
        include_i = [i for i,v in usecoefs_sig.items() if v>=self.min_sig]
        equation = ' + '.join(list(coefnum.loc[include_i] + '*' + termnames.loc[include_i]))

        self.equation = equation
        self.coefs = coefdict
        self.coefs_long = coefdict_long
        self.coefs_err = coefdict_err
        self.coefs_err_long = coefdict_long_err
        self.coefs_sig = coefdict_sig
        self.coefs_sig_long = coefdict_long_sig
    
        self.varlist = termnamesu


    def evaluate(self,sigLevel=0,**kwargs):
        if 'rm ' in self.equation:
            print('Definitely do not put rm in an eval statement')
            return
        #pdb.set_trace()
        allargs = [k for k,val in kwargs.items()]
        
        if not all(k in allargs for k in self.varlist):
        #if not all(np.sort(allargs) == self.varlist):
            print(allargs)
            print(self.varlist)
            pdb.set_trace()
            print('Inputs to evaluate must be the same as varnames')
            return

        coefs_ignore = [k for k,v in self.coefs_sig.items() if v<sigLevel]
        coefs_use = self.coefs.copy()
        if len(coefs_ignore)>0:
            for ci in coefs_ignore:
                coefs_use[ci] = 0

        locals().update(coefs_use)

        argdict = {k:val for k,val in kwargs.items()}
        locals().update(argdict)
        
        out = eval(self.equation)
        return(out)

    
    def evaluate_termbyterm(self,sigLevel=0,**kwargs):
        if 'rm ' in self.equation:
            print('Definitely do not put rm in an eval statement')
            return
        allargs = [k for k,val in kwargs.items()]
        
        if not all(k in allargs for k in self.varlist):
            print(allargs)
            print(self.varlist)
            print('Inputs to evaluate must be the same as varnames')
            return

        coefs_ignore = [k for k,v in self.coefs_sig.items() if v<sigLevel]
        coefs_use = self.coefs.copy()
        if len(coefs_ignore)>0:
            for ci in coefs_ignore:
                coefs_use[ci] = 0

        locals().update(coefs_use)

        argdict = {k:val for k,val in kwargs.items()}
        locals().update(argdict)
        
        term_list = self.equation.split(' + ')
        out = {}
        for term in term_list:
            out[term] = eval(term)
        return(out)


