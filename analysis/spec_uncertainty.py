import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
#%matplotlib qt5

run_name_ssps = ['sspgdp_gdptime_1990_quadT_linP',
                 #'sspgdp_gdptime_1990_linT_linP',
                 'sspgdp_gdptime_1990_quadT_quadP',
                 'sspgdp_gdptime_1990_quadT_quadP_TP',
                 'sspgdp_gdptime_1990_quadT_quadP_Ttrend',
                 'sspgdp_gdptime_1990_quadT_quadP_Ttrend_Tvar',
                 'sspgdp_gdptime_1990_quadT_quadP_Ttrend_Tvar_Ptrend',
                 'sspgdp_gdptime_1990_quadT_quadP_Ttrend_Tvar_Ptrend_Pvar',
                 'sspgdp_gdptime_1990_TTmean_linP',
                 #'sspgdp_gdptime_1990_quadT_linP_1990onward',
                 'sspgdp_gdptime_1990_quadT_linP_nosplit',
                 'sspgdp_gdptime_1990_quadT_linP_Tm1_inc',
                 #'sspgdp_gdptime_extrap_Ttrend_Tvar_Ptrend_Pvar',
                 ]

run_name_fixs = [r.replace('sspgdp','fixgdp') for r in run_name_ssps]

# Could potentially add messy legend
#equations = {
#    'sspgdp_gdptime_1990_quadT_linP':'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
#    'sspgdp_gdptime_1990_quadT_quadP' 
#}

projssps = {}
projfixs = {}
for run_name_ssp, run_name_fix in zip(run_name_ssps,run_name_fixs):
    short_name = run_name_ssp.replace('sspgdp_gdptime_extrap_','')
    with open('../data/projections/{}/rep_full.pickle'.format(run_name_ssp), 'rb') as handle:
        projssp = pickle.load(handle)
    projssps[short_name] = projssp
    with open('../data/projections/{}/rep_full.pickle'.format(run_name_fix), 'rb') as handle:
        projfix = pickle.load(handle)
    projfixs[short_name] = projfix

plotvars = ['mig_rate_tot','mig_num_tot_inint','mig_num_tot_outint']
plotnames = {'mig_rate_tot':'Migration rate',
             'mig_num_tot_inint':'International in-migrants',
             'mig_num_tot_outint':'International out-migrants'}
ccodes = ['USA','IND','CHN','DEU','ZAF']
#ssps = [1,2,3,5]
ssp = 2
#ccodes = projssp.adm0_country_id.unique()
ccols = {ccode:[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for ccode in ccodes}
scols = {1:'lightblue',2:'darkblue',3:'goldenrod',5:'firebrick'}
        
#yrs     = [2030,2040,2050]
#alphas  = {2030: 0.2, 2040:0.6, 2050:1.0}
#xadj    = {2030:-0.2, 2040:0.0, 2050:0.2}

yr      = 2050
specs   = list(projssps.keys())
alphas  = {spec: 1 for spec in specs}
xadj    = {spec: a  for spec,a in zip(specs,np.linspace(-0.3,0.3,len(specs)))}
xadjssp = {1:-0.06,2:-0.03,3:0.03,5:0.06}

plt.figure(figsize=(len(plotvars)*5.5,7))
for pi, pv in enumerate(plotvars):
    ax = plt.subplot(2,len(plotvars),pi+1)
    for ci,ccode in enumerate(ccodes):
        for speci,spec in enumerate(specs):
            projssp = projssps[spec]
            projfix = projfixs[spec]
            udat = []
            for rep in projssp.keys():
                udatrep = projssp[rep].loc[(projssp[rep].ssp==ssp) & 
                                           (projssp[rep].year==yr) & 
                                           (projssp[rep].adm0_country_id==ccode)]
                udat.append(udatrep)
            udat = pd.concat(udat)
            if (pi==1) & (ci==0):
                plt.plot(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],'x',
                         alpha=0,color=scols[ssp],
                         label='SSP{}, {}'.format(ssp,yr))
                plt.text(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],str(speci),
                         alpha=alphas[spec],color=scols[ssp],ha='center',va='center',
                         label='SSP{}, {}'.format(ssp,yr))
            else:
                plt.plot(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],'x',
                         alpha=0,color=scols[ssp])
                plt.text(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],str(speci),
                         alpha=alphas[spec],color=scols[ssp],ha='center',va='center',)

    #plt.legend()
    ax.set_xticks(range(len(ccodes)))
    ax.set_xticklabels(ccodes,fontsize=14)
    plt.ylabel('Change in ' + plotnames[pv])
    plt.title(plotnames[pv]+', SSP GDP')
# Repeat for fixed
for pi, pv in enumerate(plotvars):
    ax = plt.subplot(2,len(plotvars),pi+len(plotvars)+1)
    for ci,ccode in enumerate(ccodes):
        for speci,spec in enumerate(specs):
            projssp = projssps[spec]
            projfix = projfixs[spec]
            udat = []
            for rep in projfix.keys():
                udatrep = projfix[rep].loc[(projfix[rep].ssp==ssp) & 
                                           (projfix[rep].year==yr) & 
                                           (projfix[rep].adm0_country_id==ccode)]
                udat.append(udatrep)
            udat = pd.concat(udat)
            plt.plot(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],'x',
                     alpha=0,color=scols[ssp])
            plt.text(ci*np.ones(len(udat))+xadj[spec]+xadjssp[ssp],udat[pv],str(speci),
                     alpha=alphas[spec],color=scols[ssp],ha='center',va='center')
    ax.set_xticks(range(len(ccodes)))
    ax.set_xticklabels(ccodes,fontsize=14)
    plt.ylabel('Change in ' + plotnames[pv])
    plt.title(plotnames[pv]+', fixed GDP')
plt.suptitle('Specification uncertainty\nSSP{}, {}'.format(ssp,yr))
plt.tight_layout()
    
plt.savefig('../figures/specification_uncertainty.png',dpi=400)
