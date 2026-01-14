import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import norm
import xarray as xr

# Make robustness cumulative distribution and map

def z_value(conf_level):
    alpha = 1 - conf_level
    return norm.ppf(1 - alpha / 2)

def se_to_conf(se):
    return 2 * norm.cdf(se) - 1

def confidence_level_from_z(z):
    return 2 * norm.cdf(z) - 1

specmap = {'T+gdppc1':'robustness_T_gdppc1_mean.csv',
           'T+gdppc1+gdppc2':'robustness_T_gdppc1_gdppc2.csv',
           'T+gdppc1+P':'robustness_T_gdppc1_P.csv',
           }

speccols = {'T+gdppc1':'k',
            'T+gdppc1+gdppc2':'gray',
            'T+gdppc1+P':'steelblue',
            }

specnames = list(specmap.keys())

dat = {sn:pd.read_csv('../data/robustness/'+specfile) for sn, specfile in specmap.items()}

plt.figure()
ax1 = plt.subplot(111)
for sn, datu in dat.items():
    #uerr = datu['SE_error'].clip(*np.nanpercentile(np.abs(datu['SE_error']),[0,99]))
    uerr = datu['SE_error']
    sns.kdeplot(np.abs(uerr),cut=0,cumulative=True,ax=ax1,color=speccols[sn],label=sn,bw_adjust=0.01,gridsize=1000)
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim)
    ax1.set_xlabel('Grid cell regression dif from pooled regression (SE units)')
    ax1.grid('on')
plt.legend(loc=4)

ax1.set_ylabel('Cumulative density')
ax1.set_xlim([0,4])

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([0,0.99])
CIs = [0,0.5,0.66,0.80,0.9,0.95,0.99]
zs = [z_value(ci) for ci in CIs]
ax2.set_xticks(zs)
ax2.set_xticklabels(['{:.2f}'.format(ci) for ci in CIs])
ax2.set_xlabel('Confidence interval equivalent')

#plt.savefig('../figures/robustness_cumulative.png',dpi=400)

# Check overlapping
#specmap2 = {'lower':'robustness_T_gdppc1_lower.csv',
#            'mean':'robustness_T_gdppc1_mean.csv',
#            'upper':'robustness_T_gdppc1_upper.csv',
#            }
specmap2 = {'mean':'robustness_T_gdppc1_mean.csv'}

dat2 = {sn:pd.read_csv('../data/robustness/'+specfile) for sn, specfile in specmap2.items()}

passnames = []
dat2_comb = dat2['mean'].copy() # Take the first one and tack on the other SE errors to it
for sn in specmap2.keys():
    dat2_comb['SE_error_{}'.format(sn)] = dat2[sn]['SE_error']
    pass_test  = np.abs(dat2_comb['SE_error_{}'.format(sn)]) < 1.959963984540054
    dat2_comb['pass_test_{}'.format(sn)] = pass_test
    passnames.append('pass_test_{}'.format(sn))

dat2_comb['Npass'] = dat2_comb[passnames].sum(1)

dat2_comb.groupby('Npass').count()['SE_error'] / len(dat2_comb)

# Pop weighted
#dat2_comb.loc[dat2_comb['pass_test_lower'],'pop_base_mean'].sum() / dat2_comb['pop_base_mean'].sum()
dat2_comb.loc[dat2_comb['pass_test_mean'],'pop_base_mean'].sum() / dat2_comb['pop_base_mean'].sum()
#dat2_comb.loc[dat2_comb['pass_test_upper'],'pop_base_mean'].sum() / dat2_comb['pop_base_mean'].sum()

# Make figure of where the test is passed and not passed
plt.figure(figsize=(13,4.5));
dat2grpsum   = dat2_comb.groupby(['lat_mean','lon_mean'])['pass_test_mean'].sum().reset_index()
dat2grpcount = dat2_comb.groupby(['lat_mean','lon_mean'])['pass_test_mean'].count().reset_index()
dat2grpsum['pass_frac'] = dat2grpsum['pass_test_mean'] / dat2grpcount['pass_test_mean']
dat2grpsum.columns = [c.replace('_mean','') for c in dat2grpsum.columns]
dat2grpsum = dat2grpsum.set_index(['lat','lon']).to_xarray()

dat2grpsum['pass_frac'].plot(cmap='coolwarm_r')
plt.title('Fraction of points in each cell that passed test (red=bad, blue=good)')
plt.gca().set_aspect(1)
plt.tight_layout()

#plt.savefig('../figures/robustness_pass_map.png',dpi=400)

# Stats by continent
rast      = xr.open_dataset('../data/political/adminID_highres_coastlines_mig.nc')
cids      = pd.read_csv('../data/political/countrycodes.csv')
romi      = cids[cids['name']=='Romania'].index
cids.loc[romi,'alpha-3'] = 'ROM'

country2continent = cids[['alpha-3','region']]
country2continent = country2continent.set_index('alpha-3')
dat2_comb['continent_id'] = dat2_comb['adm0_country_id'].map(country2continent.to_dict()['region'])
SAi = (dat2_comb.lat_mean<13.5) & (dat2_comb.continent_id=='Americas')
dat2_comb.loc[SAi,'continent_id'] = 'South America'
Npass_cont  = dat2_comb.groupby('continent_id')['Npass'].sum()
count_cont  = dat2_comb.groupby('continent_id')['Npass'].count()
pfrac_cont  = Npass_cont/count_cont

print(pfrac_cont)
