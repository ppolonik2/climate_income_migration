import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
import sys
import namelist_run as nl
sys.path.append('{}/utils/'.format(nl.root_dir))
import utilfuncs as uf
import seaborn as sns
import pdb
import os

Nreps       = nl.Nreps
sample_frac = nl.sample_frac

# Set equations to run
equations = {nl.eq_name:nl.equation}

def tab_from_panel(pan_cell,pan_adm,pan_ctry):
    summary_df = pd.DataFrame({
        'Parameter'     :pan_adm.params,
        'Std. Err.'     :pan_adm.std_errors,
        'T-stat'        :pan_adm.tstats,
        'P-value'       :pan_adm.pvalues,
        'Lower CI'      :pan_adm.conf_int().lower,
        'Upper CI'      :pan_adm.conf_int().upper,

        'Std. Err. Cell':pan_cell.std_errors,
        'T-stat Cell'   :pan_cell.tstats,
        'P-value Cell'  :pan_cell.pvalues,
        'Lower CI Cell' :pan_cell.conf_int().lower,
        'Upper CI Cell' :pan_cell.conf_int().upper,

        'Std. Err. Ctry':pan_ctry.std_errors,
        'T-stat Ctry'   :pan_ctry.tstats,
        'P-value Ctry'  :pan_ctry.pvalues,
        'Lower CI Ctry' :pan_ctry.conf_int().lower,
        'Upper CI Ctry' :pan_ctry.conf_int().upper,

        'Nobs'          :pan_adm.nobs,
        'R2 inclusive'  :pan_adm.rsquared_inclusive,
        'R2 within'     :pan_adm.rsquared_within,
    })
    return summary_df

xvars = {
    'cubic_income':'T',
    'quadratic_income': 'T',
    'quad_inc_Ttrend_Tvar_P': 'P',
    'quad_inc_Ttrend_Tvar_P_P2': 'P',
    'quadratic_income_Tmean': 'T',
    'cubic_income_Tmean': 'T',
    'quad_inc_Ttrend_Tvar_P_Ptrend_Pvar':'T',
    'T_P_Ttrend_quadinc_linTtrend':'T',
    'T_P_Ttrend_quadinc':'T',
}
for eq_name, eq in equations.items():
    if (eq_name not in xvars):
        xvars[eq_name] = 'T'
other_opt = {'Ttrend':0,'Ttrend2':0,'Tvar':0,'T':15,'P':0,'P2':0*0,'Tmean':10,'Ptrend':0,'Pvar':0}

# Read dataset
#dat = pd.read_stata('../data/stata_output/mig_climate_lasso.dta')
if nl.no_split:
    migpath = nl.dat_dir+'/stata_output/climate_migr_analysis_halfdeg_nonFUA_long_PPrurOnly.csv'
    dat = pd.read_csv(migpath,index_col=0)
else:
    #migpath = nl.dat_dir+'/stata_output/climate_migr_analysis_halfdeg_nonFUA_long.dta'
    migpath = nl.dat_dir+'/stata_output/climate_migr_analysis_halfdeg_ALL_long.dta'
    dat = pd.read_stata(migpath)
    if nl.urban_opt:
        dat = dat.loc[dat['d_urban']==1]
    else:
        dat = dat.loc[dat['d_urban']==0]

# Rename variables
#dat = dat.rename(columns={'gridmean_gdp_pcap':'gdppc',
dat = dat.rename(columns={'gdp_per_cap':'gdppc',
                          'adm1_state_id':'admin_id',})  # Comment out for old dataset
dat = dat.rename(columns={'temp':'T','temp_q2':'T2'})
dat = dat.rename(columns={'precip':'P','precip_q2':'P2'})
dat = dat.rename(columns={'Tvar_mon':'Tvar','Tvar2_mon':'Tvar2'})
dat = dat.rename(columns={'Pvar_mon':'Pvar','Pvar2_mon':'Pvar2'})
dat = dat.rename(columns={'Ttrend_mon':'Ttrend','Ptrend_mon':'Ptrend'})
dat = dat.rename(columns={'gridmean_temp':'Tmean','gridmean_precip':'Pmean'})
dat = dat.rename(columns={'gridmean_temp2':'Tmean2'})
dat['admin_id'] =  dat['admin_id'].astype(str)

# Add T and P anom variables
dat['Tanom']  = dat['T'] - dat['Tmean']
dat['Tanom2'] = dat['Tanom']**2
dat['Panom']  = dat['P'] - dat['Pmean']
dat['Panom2'] = dat['Panom']**2

# Create outmigration and set ouliers to nan
dat['outmig_rate_jrc'] = -dat['mig_rate_jrc']
dat.loc[dat['gdppc']<=300,   'gdppc'] = np.nan
dat.loc[dat['gdppc']>=200000,'gdppc'] = np.nan
dat.loc[dat['mig_rate_jrc']<=-1,'mig_rate_jrc'] = np.nan
dat.loc[dat['mig_rate_jrc']>=2 ,'mig_rate_jrc'] = np.nan

# Log GDP and add quadratic/cubic
dat['gdppc1'] = np.log(dat['gdppc']) # Unlog for old dataset

# Extrapolation of 1990 GDP back to 1975
if nl.extrap_gdp=='linear':
    gdppiv   = dat.pivot(index=['lat','lon','admin_id'],columns='year',values='gdppc1')
    gdpslope = gdppiv[1995]-gdppiv[1990]
    gdppiv[1985] = gdppiv[1990]-gdpslope
    gdppiv[1980] = gdppiv[1990]-gdpslope*2
    gdppiv[1975] = gdppiv[1990]-gdpslope*3
    gdpextrap = gdppiv.stack()
    gdpextrap.name = 'gdppc1_extrapolate'
    dat = dat.merge(gdpextrap,on=['lat','lon','admin_id','year'],how='left')
    dat['gdppc1'] = dat['gdppc1_extrapolate']
elif nl.extrap_gdp=='constant':
    gdppiv   = dat.pivot(index=['lat','lon','admin_id'],columns='year',values='gdppc1')
    gdppiv[1985] = gdppiv[1990]
    gdppiv[1980] = gdppiv[1990]
    gdppiv[1975] = gdppiv[1990]
    gdpextrap = gdppiv.stack()
    gdpextrap.name = 'gdppc1_extrapolate'
    dat = dat.merge(gdpextrap,on=['lat','lon','admin_id','year'],how='left')
    dat['gdppc1'] = dat['gdppc1_extrapolate']

# End extrapolation

# Cut dat to remove extreme T (remove 0.2% of population)
perc_cut = 0.001
dnn   = dat[['T','pop_base']].dropna()
Tperc = uf.weighted_quantile(dnn['T'], [perc_cut,1-perc_cut],dnn['pop_base'])
dnn   = dat[['Ttrend','pop_base']].dropna()
Ttrendperc = uf.weighted_quantile(dnn['Ttrend'], [perc_cut,1-perc_cut],dnn['pop_base'])
dnn   = dat[['P','pop_base']].dropna()
Pperc = uf.weighted_quantile(dnn['P'], [0,1-perc_cut],dnn['pop_base'])
#dnn   = dat[['mig_rate_jrc','pop_base']].dropna()
#migperc = uf.weighted_quantile(dnn['mig_rate_jrc'], [perc_cut,1-perc_cut],dnn['pop_base'])

dat   = dat[(dat['T']            >Tperc[0])      & (dat['T']            < Tperc[1])]
dat   = dat[(dat['Ttrend']       >Ttrendperc[0]) & (dat['Ttrend']       < Ttrendperc[1])]
dat   = dat[(dat['P']            >=Pperc[0])     & (dat['P']            < Pperc[1])]
#dat   = dat[(dat['mig_rate_jrc'] >migperc[0])    & (dat['mig_rate_jrc'] < migperc[1])]

dnn        = dat[['gdppc1','pop_base']].dropna()
gdppc1perc = uf.weighted_quantile(dnn['gdppc1'],[0,1],dnn['pop_base'])
#dat = dat[(dat['gdppc1']>gdppc1perc[0]) & (dat['gdppc1'] < gdppc1perc[1])]

# Convert GDP to mean GDP if selected
if nl.mean_gdppc:
    gdpmean = dat.groupby(['lat','lon','admin_id'])['gdppc1'].mean()
    gdpmean.name = 'gdppc1_mean'
    dat = dat.merge(gdpmean,on=['lat','lon','admin_id'])
    dat['gdppc1'] = dat['gdppc1_mean']

# Shuffle T and P in time as sensitivity if selected
if nl.shuffle_clim:
    def deranged_permutation(row):
        """Return a shuffled version of the row that is not identical to the original."""
        arr = row.values.copy()
        while True:
            shuffled = np.random.permutation(arr)
            if not np.array_equal(shuffled, arr):
                return shuffled
    tmpT  = dat.pivot(index=['lat','lon','admin_id'],columns='year',values='T')
    pivT  = tmpT.apply(deranged_permutation,axis=1,result_type='expand')
    pivT.columns = tmpT.columns
    pivT = pivT.stack()
    pivT.name = 'T'
    tmpP  = dat.pivot(index=['lat','lon','admin_id'],columns='year',values='P')
    pivP  = tmpP.apply(np.random.permutation,axis=1,result_type='expand')
    pivP.columns = tmpP.columns
    pivP = pivP.stack()
    pivP.name = 'P'

    dat   = dat.drop(columns=['T','T2','P'])
    dat   = dat.merge(pivT.reset_index(),on=['lat','lon','admin_id','year'])
    dat   = dat.merge(pivP.reset_index(),on=['lat','lon','admin_id','year'])
    dat['T2'] = dat['T']**2

    pdb.set_trace()

# Open hist GDP and SSP gdp and merge with dat
#gdp = pd.read_csv('/home/ppolonik/../j4wan/Migration/Pop_mig/Pop_mig_data/gdp_05deg_gridmean.csv')
#gdp_fut   = xr.open_dataset('~/../j4wan/Migration/Pop_mig/Pop_mig_data/ssp_gdp/gdp_pop_ssps.nc')
#gdp_fut50 = gdp_fut.sel({'ssp':2,'year':2050})['gdppc'].drop_vars(['ssp','year']).to_dataframe()
#gdp_fut50 = gdp_fut50.rename(columns={'gdppc':'gdppc50'}).reset_index()
#gdp_fut50['gdppc50'] = np.log(gdp_fut50['gdppc50'])
##gdp = gdp.merge(gdp_fut50,on=['lat','lon'])
#dat = dat.merge(gdp_fut50,on=['lat','lon'])

# Merge SSP2 temperature and precip for histograms
#clim = xr.open_dataset(nl.dat_dir+'/climate/climate_metrics_ssp245_halfdeg_5yr.nc')
#clim = clim.sel(model='Mean',time='2050').drop_vars(['model','time']).to_dataframe()
#clim = clim.rename(columns={'Temperature':'T50','Precipitation':'P50'})[['T50','P50']].reset_index()
clim = pd.read_csv(nl.dat_dir+'/climate/ssp2_clim.csv',index_col=0)
dat = dat.merge(clim,on=['lat','lon'])

dat['gdppc2'] = dat['gdppc1']**2
dat['gdppc3'] = dat['gdppc1']**3
dat['T*Tmean']  = dat['T']*dat['Tmean']
dat['T2*Tmean'] = dat['T2']*dat['Tmean']
dat['T*Tmean*gdppc1']  = dat['T']*dat['Tmean']*dat['gdppc1']
dat['T2*Tmean*gdppc1'] = dat['T2']*dat['Tmean']*dat['gdppc1']
dat['T*Tmean*gdppc2']  = dat['T']*dat['Tmean']*dat['gdppc2']
dat['T2*Tmean*gdppc2'] = dat['T2']*dat['Tmean']*dat['gdppc2']
dat['Tanom*Tmean*gdppc1']  = dat['T']*dat['Tmean']*dat['gdppc1']
dat['Tanom2*Tmean*gdppc1'] = dat['T2']*dat['Tmean']*dat['gdppc1']
dat['Tanom*Tmean*gdppc2']  = dat['T']*dat['Tmean']*dat['gdppc2']
dat['Tanom2*Tmean*gdppc2'] = dat['T2']*dat['Tmean']*dat['gdppc2']
dat['T*P']        = dat['T']*dat['P']
dat['T*P*gdppc1'] = dat['T']*dat['P']*dat['gdppc1']
dat['T*P*gdppc2'] = dat['T']*dat['P']*dat['gdppc2']

# Add lags
Tm1      =  dat.pivot(index=['lat','lon','admin_id'],columns='year',values='T').shift(1,axis=1).stack()
Pm1      =  dat.pivot(index=['lat','lon','admin_id'],columns='year',values='P').shift(1,axis=1).stack()
outmigm1 =  dat.pivot(index=['lat','lon','admin_id'],columns='year',values='outmig_rate_jrc').shift(1,axis=1).stack()
Tm1.name      = 'Tm1'
Pm1.name      = 'Pm1'
outmigm1.name = 'outmigm1'
dat = dat.merge(Tm1,     on=['lat','lon','admin_id','year'])
dat = dat.merge(Pm1,     on=['lat','lon','admin_id','year'])
dat = dat.merge(outmigm1,on=['lat','lon','admin_id','year'])
dat['T2m1']        = dat['Tm1']**2
dat['P2m1']        = dat['Pm1']**2
dat['Tm1*gdppc1']  = dat['Tm1']*dat['gdppc1']
dat['Tm1*gdppc2']  = dat['Tm1']*dat['gdppc2']
dat['Pm1*gdppc1']  = dat['Pm1']*dat['gdppc1']
dat['Pm1*gdppc2']  = dat['Pm1']*dat['gdppc2']
dat['T2m1*gdppc1'] = dat['T2m1']*dat['gdppc1']
dat['T2m1*gdppc2'] = dat['T2m1']*dat['gdppc2']
# Add income interaction variables
for v in ['T','P','Ttrend','Tvar','Tmean','Ptrend','Pvar','Tanom','Panom']:
    dat['{}*gdppc1'.format(v)]  = dat['{}'.format(v)]  * dat['gdppc1']
    dat['{}*gdppc2'.format(v)]  = dat['{}'.format(v)]  * dat['gdppc2']
    #dat['{}*gdppc3'.format(v)]  = dat['{}'.format(v)]  * dat['gdppc3']
    if v+'2' in dat.columns:
        dat['{}2*gdppc1'.format(v)] = dat['{}2'.format(v)] * dat['gdppc1']
        dat['{}2*gdppc2'.format(v)] = dat['{}2'.format(v)] * dat['gdppc2']
        #dat['{}2*gdppc3'.format(v)] = dat['{}2'.format(v)] * dat['gdppc3']
dat['constant']  = 1

# Add entry ID
tmp = dat[['lat','lon','admin_id','year']].groupby(['lat','lon','admin_id'])[['year']].count()
entryids = range(len(tmp))
tmp['entry_id'] = entryids
dat = dat.merge(tmp['entry_id'].reset_index(),on=['lat','lon','admin_id'])

# Option to limit analysis to 1990 onward
if nl.onward_1990:
    dat = dat.loc[dat.year>=1990]
    #dat = dat.loc[dat.year<1990]

if nl.richthird:
    topthird = np.nanpercentile(dat['gridmean_gdp_pcap'],66.66)
    dat = dat.loc[dat.gridmean_gdp_pcap>topthird]

# Add cubic terms
dat['T3']        = dat['T']**3
dat['T3*gdppc1'] = dat['T3']*dat['gdppc1']
dat['T3*gdppc2'] = dat['T3']*dat['gdppc2']

dat = dat.set_index(['entry_id','year'])


# This is the dataset that is actually used. Uncomment to save.
#  Needed for robustness test
if nl.save_everything:
   dat.to_csv('regres_dat.csv')

for rep in range(Nreps):
    if nl.sample_type=='random':
        # Randomly sample for error estimation
        sample_entries = pd.DataFrame(dat.index.get_level_values('entry_id')).drop_duplicates()
        sample_entries = sample_entries.sample(frac=sample_frac)
        datsamp = dat.loc[sample_entries['entry_id']]
    if nl.sample_type=='admin':
        pairs = dat.reset_index()[['entry_id','admin_id']].drop_duplicates()
        sample_entries = pairs.groupby('admin_id').sample(n=1)
        datsamp = dat.loc[sample_entries['entry_id']]

    # If no_weights is true, set population to 1 everywhere, just used for regressions
    if nl.no_weights:
        datsamp['pop_base'] = 1
    
    # Run regressions
    panels = {}
    for eq_name, equation in equations.items():
        outpath = '{}/reg/'.format(nl.out_dir)
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        panel = PanelOLS(datsamp['outmig_rate_jrc'],datsamp[equation.split('+')],
                         entity_effects=True,time_effects=True,drop_absorbed=True,
                         weights=datsamp['pop_base']).fit(
                         #cov_type='clustered',cluster_entity=True)
                         cov_type='clustered',clusters=datsamp['admin_id'])
        panels[eq_name] = panel

        panel_cellcluster = PanelOLS(datsamp['outmig_rate_jrc'],datsamp[equation.split('+')],
                         entity_effects=True,time_effects=True,drop_absorbed=True,
                         weights=datsamp['pop_base']).fit(
                         cov_type='clustered',cluster_entity=True)

        panel_ctrycluster = PanelOLS(datsamp['outmig_rate_jrc'],datsamp[equation.split('+')],
                         entity_effects=True,time_effects=True,drop_absorbed=True,
                         weights=datsamp['pop_base']).fit(
                         cov_type='clustered',clusters=datsamp['adm0_country_id'])
    
        paneltab = tab_from_panel(panel_cellcluster,panel,panel_ctrycluster)
        datsampu = datsamp[equation.split('+')+['outmig_rate_jrc']].dropna()
        paneltab['mean dev var'] = np.mean(datsampu['outmig_rate_jrc'])
        paneltab['sd dev var']   = np.std(datsampu['outmig_rate_jrc'])

        #pdb.set_trace()

        paneltab.to_csv(outpath+'rep{}.csv'.format(str(rep).zfill(3)))
        panel.cov.to_csv(outpath+'rep{}_cov.csv'.format(str(rep).zfill(3)))

