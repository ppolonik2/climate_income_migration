import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
import seaborn as sns
import os

gdp_opt = 'mean' # Whether to use 'lower', 'mean', or 'upper' GDP data point to use in comparison

dat = pd.read_csv('../data/regres/regres_dat.csv',index_col=[0,1])

equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'

panel = PanelOLS(dat['outmig_rate_jrc'],dat[equation.split('+')],
                 entity_effects=True,time_effects=True,drop_absorbed=True,
                 weights=dat['pop_base']).fit(
                 cov_type='clustered',clusters=dat['admin_id'])

panel_justT = PanelOLS(dat['outmig_rate_jrc'],dat[['T','T2','constant']],
                       entity_effects=True,time_effects=True,drop_absorbed=True,
                       weights=dat['pop_base']).fit(
                       cov_type='clustered',clusters=dat['admin_id'])

panel_unitcluster = PanelOLS(dat['outmig_rate_jrc'],dat[equation.split('+')],
                             entity_effects=True,time_effects=True,drop_absorbed=True,
                             weights=dat['pop_base']).fit(
                             cov_type='clustered')

panel_countrycluster = PanelOLS(dat['outmig_rate_jrc'],dat[equation.split('+')],
                                entity_effects=True,time_effects=True,drop_absorbed=True,
                                weights=dat['pop_base']).fit(
                                cov_type='clustered',clusters=dat['adm0_country_id'])

# Make dataframe that summarizes results from different clustering
c_adm1 = panel.params.to_frame(name='admin1')
c_ctry = panel_countrycluster.params.to_frame(name='country')
c_unit = panel_unitcluster.params.to_frame(name='unit')

e_adm1 = panel.std_errors.to_frame(name='admin1')
e_ctry = panel_countrycluster.std_errors.to_frame(name='country')
e_unit = panel_unitcluster.std_errors.to_frame(name='unit')

p_adm1 = panel.pvalues.to_frame(name='admin1')
p_ctry = panel_countrycluster.pvalues.to_frame(name='country')
p_unit = panel_unitcluster.pvalues.to_frame(name='unit')

panel_cluster_c = pd.concat([c_adm1,c_ctry,c_unit],axis=1)
panel_cluster_e = pd.concat([e_adm1,e_ctry,e_unit],axis=1)
panel_cluster_p = pd.concat([p_adm1,p_ctry,p_unit],axis=1)

panel_cluster_c.columns = pd.MultiIndex.from_product([['Parameter'], panel_cluster_c.columns])
panel_cluster_e.columns = pd.MultiIndex.from_product([['Std Error'], panel_cluster_e.columns])
panel_cluster_p.columns = pd.MultiIndex.from_product([['p-value'], panel_cluster_p.columns])

panel_cluster_summary = pd.concat([panel_cluster_c,panel_cluster_e,panel_cluster_p],axis=1)

# End clustering summary

def tab_from_panel(pan):
    summary_df = pd.DataFrame({
        'Parameter':pan.params,
        'Std. Err.':pan.std_errors,
        'T-stat'   :pan.tstats,
        'P-value'  :pan.pvalues,
        'Lower CI' :pan.conf_int().lower,
        'Upper CI' :pan.conf_int().upper
    })
    return summary_df

paneltab = tab_from_panel(panel)

# Run through every entry_id and run a linear regression 
# see what fraction overlaps with main panel above evaluated at the mean GDPpc

eids = dat.reset_index()['entry_id'].unique()

gridslopes = pd.DataFrame(index=eids,columns=['T','T_se'])

if not os.path.exists('robustness_T_gdppc1.csv'):
    for i, eid in enumerate(eids):
        if i % 1000 == 0: print(i)
        #griddat = dat.loc[eid][['outmig_rate_jrc','T','gdppc1','gdppc2']].dropna()
        griddat = dat.loc[eid][['outmig_rate_jrc','T','gdppc1']].dropna()
        if len(griddat)>4:
            #gridreg = smf.ols(formula='outmig_rate_jrc~T+gdppc1+gdppc2',data=griddat).fit()
            gridreg = smf.ols(formula='outmig_rate_jrc~T+gdppc1',data=griddat).fit()
            gridslopes.loc[eid,'T']    = gridreg.params.loc['T']
            gridslopes.loc[eid,'T_se'] = gridreg.bse.loc['T']
            gridslopes.loc[eid,'T_lo'] = gridreg.conf_int().loc['T',0]
            gridslopes.loc[eid,'T_up'] = gridreg.conf_int().loc['T',1]
    gridslopes.index.name = 'entry_id'

else:
    gridslopes = pd.read_csv('./robustness_T_gdppc1.csv',index_col=0)

# If running just bottom section, reload
# gridslopes = pd.read_csv('robustness_T_gdppc1.csv',index_col=0)
# Add column for the pooled result evaluated at the grid
means = dat.groupby('entry_id')[['gdppc1','gdppc2','T','pop_base','lat','lon']].mean()
means.columns = [c+'_mean' for c in means.columns]

if gdp_opt == 'upper':
    means['gdppc1_mean'] = dat.groupby('entry_id')['gdppc1'].max()
elif gdp_opt=='lower':
    means['gdppc1_mean'] = dat.groupby('entry_id')['gdppc1'].min()

gridslopes = gridslopes.merge(means,on='entry_id')

p = paneltab['Parameter']
gridslopes['T_eval']=(p.loc['T']                                                                 + 
                      p.loc['T*gdppc1']      * gridslopes['gdppc1_mean']                         + 
                      p.loc['T*gdppc2']      * gridslopes['gdppc1_mean']**2                      + 
                      p.loc['T2']        * 2                               * gridslopes['T_mean'] + 
                      p.loc['T2*gdppc1'] * 2 * gridslopes['gdppc1_mean']   * gridslopes['T_mean'] + 
                      p.loc['T2*gdppc2'] * 2 * gridslopes['gdppc1_mean']**2* gridslopes['T_mean']) 

gridslopes = gridslopes.dropna()

Ngrid = len(gridslopes.dropna())
pass_test      = (( gridslopes['T_eval'] < gridslopes['T_up'] ) &
                  ( gridslopes['T_eval'] > gridslopes['T_lo'] ))

pass_test_frac = np.sum(pass_test)/Ngrid

testfracpopT = gridslopes['pop_base_mean'].loc[pass_test].sum() / gridslopes['pop_base_mean'].sum()

# How many standard errors away is global dT
gridslopes['dif_bw_grid_eval'] = gridslopes['T'] - gridslopes['T_eval']
gridslopes['SE_error']         = gridslopes['dif_bw_grid_eval'] / gridslopes['T_se']

# Add back country and admin IDs
ids = dat.reset_index()[['entry_id','adm0_country_id','admin_id']].drop_duplicates()
gridslopes = gridslopes.merge(ids,on='entry_id',how='left')

gridslopes.to_csv('../data/robustness/robustness_T_gdppc1_{}.csv'.format(gdp_opt))


