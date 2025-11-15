import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import seaborn as sns
from scipy.stats import gaussian_kde
import pdb
#%matplotlib qt5

run_names = ['gdptime_1990_quadT_linP_rep100_50perc',
             'gdptime_1990_quadT_linP_allmodels']
            
run_easynames = {'gdptime_1990_quadT_linP_statrep100':'Statistical',
                 'gdptime_1990_quadT_linP_rep100_50perc':'Statistical',
                 'gdptime_1990_quadT_linP_rural_adminsamp':'Statistical',
                 'gdptime_1990_quadT_linP_allmodels':'Climate',
                 'gdptime_1990_quadT_linP_rep100':'Both'}

plotvars  = ['mig_rate_tot','mig_num_tot_inint','mig_num_tot_outint']
ccodes    = ['USA','IND','CHN','DEU','ZAF','ETH']
ccodes2   = [['NGA','ETH','ZAF','KEN'],
             ['JPN','SAU','IND','CHN','AUS','VNM'],
             ['GRC','RUS','ESP','DEU'],
             ['USA','MEX','PAN'],
             ['CHL','ARG','BOL','BRA']]
regions   = ['Africa','Asia-Pacific','Europe','North America','South America']
ssps      = [1,2,3,5]
reload    = False
save      = False

plotvars_easy = {'mig_rate_tot':'$\Delta$ Migration Rate',
                 'mig_num_tot_inint':'$\Delta$ International In-migrants [Mil]',
                 'mig_num_tot_outint':'$\Delta$ International Out-migrants [Mil]'}
keepcols = ['year','adm0_country_id','ssp','model']+plotvars
dfssps = {}
dffixs = {}

def weighted_violin(ax,x,vals,wts,width=0.4,lab='',col='k',hatch=None,cut95=True):
    # Weighted KDE

    # 95% of data points 
    if cut95:
        outlims = np.nanpercentile(vals,(2.5,97.5))
        include = (vals>outlims[0]) & (vals<outlims[1])
        vals = vals[include]
        if wts:
            wts = wts[include]

    kde = gaussian_kde(vals, weights=wts)
    y = np.linspace(vals.min(), vals.max(), 500)
    density = kde(y)

    # Normalize width
    density = density / density.max() * width

    # Mirror the density to make a violin shape
    ax.fill_betweenx(y, x - density, x + density, label=lab,color=col,hatch=hatch,alpha=0.6)

if reload:
    for run_name in run_names:
        run_name_ssp = 'sspgdp_'+run_name
        run_name_fix = 'fixgdp_'+run_name

        # Load full projection and extract what we need
        with open('../data/projections/{}/rep_full.pickle'.format(run_name_ssp), 'rb') as handle:
            proj = pickle.load(handle)

        # Extract data from ssp 
        dfssps[run_name] = pd.DataFrame([])
        for i,df in proj.items():
            tmpdf = df[keepcols]
            #tmpdf = df.loc[df.adm0_country_id.isin(ccodes)]
            tmpdf['rep'] = i

            dfssps[run_name] = pd.concat([dfssps[run_name],tmpdf])

        # Repeat for fixed-climate projection
        with open('../data/projections/{}/rep_full.pickle'.format(run_name_fix), 'rb') as handle:
            proj = pickle.load(handle)
    
        # Extract data from fix
        dffixs[run_name] = pd.DataFrame([])
        for i,df in proj.items():
            tmpdf = df[keepcols]
            #tmpdf = df.loc[df.adm0_country_id.isin(ccodes)]
            tmpdf['rep'] = i
            dffixs[run_name] = pd.concat([dffixs[run_name],tmpdf])

    if save:
        for run_name in run_names:
            dfssps[run_name].to_csv('uncertainties_ssp_{}.csv'.format(run_name))
            dffixs[run_name].to_csv('uncertainties_fix_{}.csv'.format(run_name))

else:
    for run_name in run_names:
        dfssps[run_name] = pd.read_csv('uncertainties_ssp_{}.csv'.format(run_name),index_col=0)
        dffixs[run_name] = pd.read_csv('uncertainties_fix_{}.csv'.format(run_name),index_col=0)

# Convert international counts to millions
for run_name in run_names:
    intcols = [c for c in dfssps[run_name].columns if 'num' in c]
    dfssps[run_name][intcols] /= 1e6
    dffixs[run_name][intcols] /= 1e6

ccols = {ccode:[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for ccode in ccodes}
scols = {1:'#F5A560',2:'#F57F73',3:'#ED1E24',5:'#8A171A'}
shats = {0:'//////',1:'\\\\\\\\\\\\',2:'xxxxxx'} 
        
yr      = 2050
alphas  = {2030: 0.2, 2040:0.6, 2050:1.0}
xadj    = {2030:-0.2, 2040:0.0, 2050:0.2}
xadjssp = {1:-0.06,2:-0.03,3:0.03,5:0.06}

plt.figure(figsize=(len(plotvars)*5,7))
dx0    = -0.45
spacer = (np.abs(dx0)*2)/(len(run_names)*len(ssps))
for pi, pv in enumerate(plotvars):
    ax = plt.subplot(2,len(plotvars),pi+1)
    for ci,ccode in enumerate(ccodes):
        dx = dx0 + spacer/2
        for si,ssp in enumerate(ssps):
            for ri,rn in enumerate(run_names):
                udat = dfssps[rn].loc[(dfssps[rn].ssp==ssp) & 
                                      (dfssps[rn].year==yr) & 
                                      (dfssps[rn].adm0_country_id==ccode)]

                if ci==0:
                    weighted_violin(ax,x=ci+dx,vals=udat[pv],
                                    wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri],
                                    lab='SSP{}, {}'.format(ssp,run_easynames[rn]))
                else:
                    weighted_violin(ax,x=ci+dx,vals=udat[pv],
                                    wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri])

                dx += spacer

    if pi==0:
        plt.ylabel('Climate + Income',fontsize=14)

    if pi==(len(plotvars)-1):
        plt.legend()
    plt.title(plotvars_easy[pv],fontsize=14)
    ax.set_xticks(range(len(ccodes)))
    ax.set_xticklabels(ccodes,fontsize=14)
    vlinex = (np.arange(len(ccodes))+0.5)[:-1]
    ylim = ax.get_ylim()
    xlim = (-0.5,len(ccodes)-0.5)
    ax.vlines(vlinex,ylim[0],ylim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax.hlines(0,xlim[0],xlim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

# Repeat for fixed
for pi, pv in enumerate(plotvars):
    ax = plt.subplot(2,len(plotvars),pi+1+len(plotvars))
    for ci,ccode in enumerate(ccodes):
        dx = dx0 + spacer/2
        for si,ssp in enumerate(ssps):
            for ri,rn in enumerate(run_names):
                udat = dffixs[rn].loc[(dffixs[rn].ssp==ssp) & 
                                      (dffixs[rn].year==yr) & 
                                      (dffixs[rn].adm0_country_id==ccode)]
                if (pi==1) & (ci==0):
                    weighted_violin(ax,x=ci+dx,vals=udat[pv],
                                    wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri],
                                    lab='SSP{}, {}'.format(ssp,run_easynames[rn]))
                else:
                    weighted_violin(ax,x=ci+dx,vals=udat[pv],
                                    wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri])

                dx += spacer

    if pi==0:
        plt.ylabel('Climate only\n(fixed income)',fontsize=14)

    ax.set_xticks(range(len(ccodes)))
    ax.set_xticklabels(ccodes,fontsize=14)
    ylim = ax.get_ylim()
    xlim = (-0.5,len(ccodes)-0.5)
    ax.vlines(vlinex,ylim[0],ylim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax.hlines(0,xlim[0],xlim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
plt.tight_layout()
    

# Second figure which has just rates and many more countries
plt.figure(figsize=(14,7.5))
dx0    = -0.45
spacer = (np.abs(dx0)*2)/(len(run_names)*len(ssps))
ccodes2flat = []
for pi, pv in enumerate(['mig_rate_tot']):
    ax2 = plt.subplot(2,1,2)

    # Get values once to figure out plotting order using SSP2
    orders = {}
    for cir,ccoder in enumerate(ccodes2):
        midval = []
        for cir2,ccode in enumerate(ccoder):
            udat = dfssps[rn].loc[(dfssps[rn].ssp==2) & 
                                  (dfssps[rn].year==yr) & 
                                  (dfssps[rn].adm0_country_id==ccode)]
            midval.append(udat['mig_rate_tot'].mean())
        orders[cir] = np.argsort(midval)
    ci=-1
    for cir,ccoder in enumerate(ccodes2):
        for cir2,ccode in enumerate(np.array(ccoder)[orders[cir]]):
            ccodes2flat.append(ccode)
            ci+=1
            dx = dx0 + spacer/2
            for si,ssp in enumerate(ssps):
                for ri,rn in enumerate(run_names):
                    udat = dfssps[rn].loc[(dfssps[rn].ssp==ssp) & 
                                          (dfssps[rn].year==yr) & 
                                          (dfssps[rn].adm0_country_id==ccode)]
    
                    if ci==0:
                        weighted_violin(ax2,x=ci+dx,vals=udat[pv],
                                        wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri],
                                        lab='SSP{}, {}'.format(ssp,run_easynames[rn]))
                    else:
                        weighted_violin(ax2,x=ci+dx,vals=udat[pv],
                                        wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri])
    
                    dx += spacer

    if pi==0:
        plt.ylabel('Climate + Income\n'+plotvars_easy[pv],fontsize=14)

    ax2.set_xticks(range(len(ccodes2flat)))
    ax2.set_xticklabels(ccodes2flat,fontsize=12,rotation=45,ha='right')
    vlinex = (np.arange(len(ccodes2flat))+0.5)[:-1]
    ylim = ax2.get_ylim()
    xlim = (-0.5,ci-0.5)
    ax2.vlines(vlinex,ylim[0],ylim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    catlines = np.cumsum([len(x) for x in ccodes2])-0.5
    ax2.vlines(catlines,ylim[0],ylim[1],alpha=1,color='k',lw=1)
    ax2.hlines(0,xlim[0],xlim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)
    for xi,x in enumerate([-0.5]+list(catlines[:-1])):
        ax2.annotate(regions[xi],(x+0.15,ylim[0]+np.diff(ylim)*0.89),
                    xycoords='data',fontsize=13,color=[0.2]*3)

# Repeat for fixed
for pi, pv in enumerate(['mig_rate_tot']):
    ax1 = plt.subplot(2,1,1)
    orders = {}
    for cir,ccoder in enumerate(ccodes2):
        midval = []
        for cir2,ccode in enumerate(ccoder):
            udat = dfssps[rn].loc[(dffixs[rn].ssp==2) & 
                                  (dffixs[rn].year==yr) & 
                                  (dffixs[rn].adm0_country_id==ccode)]
            midval.append(udat['mig_rate_tot'].mean())
        orders[cir] = np.argsort(midval)
    ci=-1
    for cir,ccoder in enumerate(ccodes2):
        for cir2,ccode in enumerate(np.array(ccoder)[orders[cir]]):
            ci+=1
            dx = dx0 + spacer/2
            for si,ssp in enumerate(ssps):
                for ri,rn in enumerate(run_names):
                    udat = dffixs[rn].loc[(dffixs[rn].ssp==ssp) & 
                                          (dffixs[rn].year==yr) & 
                                          (dffixs[rn].adm0_country_id==ccode)]
                    if (pi==0) & (ci==0):
                        weighted_violin(ax1,x=ci+dx,vals=udat[pv],
                                        wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri],
                                        lab='SSP{}, {}'.format(ssp,run_easynames[rn]))
                    else:
                        weighted_violin(ax1,x=ci+dx,vals=udat[pv],
                                        wts=None,width=spacer*0.4,col=scols[ssp],hatch=shats[ri])

                    dx += spacer

    if pi==0:
        plt.ylabel('Climate only\n'+plotvars_easy[pv],fontsize=14)
        plt.legend(loc='upper right',bbox_to_anchor=(catlines[0]/catlines[-1]+0.5/len(ccodes2flat),1))


    ax1.set_xticks(range(len(ccodes2flat)))
    ax1.set_xticklabels(ccodes2flat,fontsize=12,rotation=45,ha='right')
    ylim = ax1.get_ylim()
    xlim = (-0.5,ci-0.5)
    ax1.vlines(vlinex,ylim[0],ylim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    catlines = np.cumsum([len(x) for x in ccodes2])-0.5
    ax1.vlines(catlines,ylim[0],ylim[1],alpha=1,color='k',lw=1)
    ax1.hlines(0,xlim[0],xlim[1],linestyle='dashed',alpha=0.5,color='k',lw=0.5)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)

    for xi,x in enumerate([-0.5]+list(catlines[:-1])):
        ax1.annotate(regions[xi],(x+0.15,ylim[0]+np.diff(ylim)*0.89),
                    xycoords='data',fontsize=13,color=[0.2]*3)

# Add shading to ax2 based on ax1 limits
ax2.fill_between(ax2.get_xlim(),ax1.get_ylim()[0],ax1.get_ylim()[1],
                 alpha=1,color='#ECF6FF',zorder=0)

# Add A and B labels
ax1.annotate('A',(0,1.01),xycoords='axes fraction',fontsize=14)
ax2.annotate('B',(0,1.01),xycoords='axes fraction',fontsize=14)

plt.subplots_adjust(left=0.1, bottom=0.07, right=1-0.02, top=1-0.05, wspace=0.07, hspace=0.25)
    
plt.savefig('../figures/projection_uncertainty.png',dpi=500)

