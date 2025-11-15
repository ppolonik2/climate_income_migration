import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

adminsampdir = '../data/projections/sspgdp_gdptime_1990_quadT_linP_rural_adminsamp/'
admincoefs = {}
for n in range(100):
    coefs_ssp_rural = pd.read_csv(
        '{}/reg/rep{}.csv'.format(adminsampdir,str(n).zfill(3)),
        index_col=0
    )
    admincoefs[n] = coefs_ssp_rural


randsampdir = '../data/projections/sspgdp_gdptime_1990_quadT_linP_statrep100/'
randcoefs = {}
for n in range(100):
    coefs_ssp_rural = pd.read_csv(
        '{}/reg/rep{}.csv'.format(randsampdir,str(n).zfill(3)),
        index_col=0
    )
    randcoefs[n] = coefs_ssp_rural

randsampdir = '../data/projections/sspgdp_gdptime_1990_quadT_linP_rep100_50perc/'
rand50coefs = {}
for n in range(100):
    coefs_ssp_rural = pd.read_csv(
        '{}/reg/rep{}.csv'.format(randsampdir,str(n).zfill(3)),
        index_col=0
    )
    rand50coefs[n] = coefs_ssp_rural

vss = [['T','T*gdppc1','T*gdppc2'],['T2','T2*gdppc1','T2*gdppc2'],['P','P*gdppc1','P*gdppc2']]
sct=0
plt.figure(figsize=(12,8))
for vs in vss:
    for v in vs:
        sct+=1
        ax = plt.subplot(len(vss),len(vs),sct)
        admincs  = [admincoefs[n].loc[v,'Parameter'] for n in range(100)]
        randcs   = [randcoefs[n].loc[v,'Parameter'] for n in range(100)]
        rand50cs = [rand50coefs[n].loc[v,'Parameter'] for n in range(100)]
        sns.kdeplot(x=admincs,color='gray',ax=ax,label='Admin1 sampling')
        sns.kdeplot(x=randcs,color='orange',ax=ax,label='85% random sample')
        sns.kdeplot(x=rand50cs,color='maroon',ax=ax,label='50% random sample')
        ax.set_xlabel(v)
        ylim = ax.get_ylim()
        ax.plot([0,0],ylim,'--k',alpha=0.3)
        ax.set_ylim(ylim)
        if sct==1:
            plt.legend()
plt.tight_layout()
        
plt.savefig('../figures/monte_carlo_coefficient_distributions.png',dpi=400)
