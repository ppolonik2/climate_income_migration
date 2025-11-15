import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.formula.api as smf


dat = pd.read_csv('../data/regres/regres_dat.csv',index_col=[0,1])
dat['const'] = 1

panel_gdppc1 = PanelOLS.from_formula('gdppc1~const+EntityEffects+TimeEffects',data=dat,
                 weights=dat['pop_base']).fit(
                 cov_type='clustered',clusters=dat['admin_id'])

panel_mig    = PanelOLS.from_formula('mig_rate_jrc~const+EntityEffects+TimeEffects',data=dat,
                 weights=dat['pop_base']).fit(
                 cov_type='clustered',clusters=dat['admin_id'])

panel_T      = PanelOLS.from_formula('T~const+EntityEffects+TimeEffects',data=dat,
                 weights=dat['pop_base']).fit(
                 cov_type='clustered',clusters=dat['admin_id'])

resid_gdp = panel_gdppc1.resids
resid_mig = panel_mig.resids
resid_T   = panel_T.resids

resid_gdp.name = 'gdp'
resid_mig.name = 'mig'
resid_T.name = 'T'

resids = pd.DataFrame(resid_gdp).merge(resid_mig,left_index=True,right_index=True)
resids = resids.merge(resid_T,left_index=True,right_index=True)

plt.figure()
plt.hist2d(resids['gdp'],resids['mig'],bins=100,norm='log')
plt.xlabel('log(GDPpc) residual')
plt.ylabel('Migration rate residual')
plt.colorbar()
plt.tight_layout()
#plt.savefig('../figures/mig_gdp_resid_hist2d.png',dpi=400)

plt.figure()
plt.hist2d(resids['T'],resids['mig'],bins=100,norm='log')
plt.xlabel('T residual')
plt.ylabel('migration residual')

res_perc_gdp = np.percentile(resids['gdp'],[5,95])
res_perc_mig = np.percentile(resids['mig'],[5,95])

residscut  = resids[(resids.gdp>res_perc_gdp[0]) & 
                    (resids.gdp<res_perc_gdp[1]) & 
                    (resids.mig>res_perc_mig[0]) & 
                    (resids.mig<res_perc_mig[1])]

plt.figure()
plt.hist2d(residscut['gdp'],residscut['mig'],bins=100,norm='log')
plt.xlabel('gdp residual')
plt.ylabel('migration residual')

resfit    = smf.ols(formula='mig~gdp',data=resids.reset_index()).fit(
                cov_type='cluster', cov_kwds={'groups': resids.reset_index()['entry_id']})
rescutfit = smf.ols(formula='mig~gdp',data=residscut.reset_index()).fit(
                cov_type='cluster', cov_kwds={'groups': residscut.reset_index()['entry_id']})

# Alternative figure with binning

bins       = np.linspace(resids['gdp'].min(),resids['gdp'].max(),50+1)
binmid     = bins[:-1]+np.diff(bins)
binymean   = np.zeros(len(binmid))
binymedian = np.zeros(len(binmid))
binN       = np.zeros(len(binmid))
for bi, (bmin,bmax) in enumerate(zip(bins[:-1],bins[1:])):
    subset         = resids[(resids['gdp']>bmin)&(resids['gdp']<bmax)]
    binymean[bi]   = subset['mig'].mean() 
    binymedian[bi] = subset['mig'].median() 
    binN[bi]       = len(subset)
    
ui = binN>10
plt.figure(figsize=(13,5))
plt.subplot(121)
plt.scatter(binmid[ui],binymean[ui],  s=binN[ui]/binN.max()*200,facecolor='none',edgecolors='firebrick')
plt.title('Bin mean')
plt.xlabel('gdp residual')
plt.ylabel('migration residual')
plt.subplot(122)
plt.scatter(binmid[ui],binymedian[ui],s=binN[ui]/binN.max()*200,facecolor='none',edgecolors='firebrick')
plt.title('Bin median')
plt.xlabel('gdp residual')
plt.ylabel('migration residual')
plt.tight_layout()



