import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# Script shortened from original file: compile_reg_coefs

runs = ['sspgdp_gdptime_1990_quadT_linP',]

#dropv = ['gdppc1','gdppc2','outmigm1','constant']
dropv = ['gdppc1','gdppc2','constant']
dats = {}
for r in runs:
    tmpdat = pd.read_csv('../data/projections/'+r+'/reg/rep000.csv',index_col=0)
    dats[r] = tmpdat[['Parameter','Std. Err.','P-value']]
    dats[r] = dats[r].drop([v for v in dropv if v in tmpdat.index])
dat = pd.concat(dats,axis=1)

#   #   #   #   #   #   #   #   #
#   Make marginal figure
#   #   #   #   #   #   #   #   #

r = 'sspgdp_gdptime_1990_quadT_linP'
cov = pd.read_csv('../data/projections/'+r+'/reg/rep000_cov.csv',index_col=0)
cov = cov.drop(dropv).drop(columns=dropv)

if 'richthird' in r:
    gdppcs = np.arange(9.7,11,step=0.3)
    lloc = 3
else:
    gdppcs = [6,7,8,9,10,11]
    lloc = 2
alpha = 0.05


#  #  #  #  Histogram curves  #  #  #  #
if True:
    # Prep for histograms
    try:
        histdat = pd.read_csv('../data/projections/{}/{}_full_05deg.csv'.format(r,r),index_col=0)
    except:
        print('WARNING: Loading historical data for a different run than the one listed')
        print('SHOULD be ok, but no guarantees')
        histdat = pd.read_csv('../data/projections/sspgdp_gdptime_1990_quadT_linP/sspgdp_gdptime_1990_quadT_linP_full_05deg.csv'.format(r,r),index_col=0)

    histdat = histdat.loc[(histdat.year==2015)|(histdat.year==2050)&(histdat.ssp==2)]
    histdat['gdppc'] = histdat['gdppc'].clip(5,13)

    # Potentially calculate and designate weighting
    histdat['gridweight'] = histdat['pop_total_rural']
    weightvar = 'gridweight' # pop_total_rural
    
    histT15 = histdat.loc[histdat.year==2015][['Temperature',  'gdppc',weightvar]]
    histP15 = histdat.loc[histdat.year==2015][['Precipitation','gdppc',weightvar]]
    Tall = np.linspace(0,32,200)
    linesT = []
    labsT = []

    plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    for gi,g in enumerate(gdppcs):
        X_pred = pd.DataFrame({'T':Tall,'T2':Tall**2,
                               'T*gdppc1':Tall*g,'T2*gdppc1':Tall**2*g,
                               'T*gdppc2':Tall*g**2,'T2*gdppc2':Tall**2*g**2,
                               'P':5,'P*gdppc1':5*g,'P*gdppc2':5*g**2})

        X_pred = X_pred[dats[r].index]
        y_hat = X_pred.values @ dats[r]['Parameter'].values

        lc_ls = []
        lc_fs = []
        for ti in range(len(Tall))[1:-1]:
            gfrac = histT15[(histT15.gdppc>(g-0.5)) &
                            (histT15.gdppc<(g+0.5)) &
                            (histT15.Temperature>Tall[ti-1]) &
                            (histT15.Temperature<Tall[ti+1])
                            ]
            gfrac = gfrac[weightvar].sum()/histT15[weightvar].sum()*500
            col = plt.cm.copper((g-np.min(gdppcs))/7)
            lc_ls.append([(Tall[ti-1],y_hat[ti-1]),(Tall[ti],y_hat[ti])])
            lc_fs.append(gfrac)

        lc = LineCollection(lc_ls,lw=lc_fs,color=col,edgecolor=col,antialiased=False)
        ax.add_collection(lc)
        linesT.append(Line2D([0],[0],color=col,linewidth=3))
        labsT.append('GDPpc={}'.format(g))
    ax.set_ylabel('Climate-driven\nrural out-migration rate',fontsize=12)
    ax.set_xlabel('Temperature',fontsize=12)
    ax.set_ylim(-0.05,0.27)
    ax.legend(linesT,labsT,loc=4)
    ax.set_xlim((0,32))

    # Repeat for precip
    ax = plt.subplot(122)
    
    Pall = np.linspace(0,30,200)
    linesP = []
    labsP = []
    for gi,g in enumerate(gdppcs):
        X_pred = pd.DataFrame({'T':10,'T2':10**2,
                               'T*gdppc1':10*g,'T2*gdppc1':10**2*g,
                               'T*gdppc2':10*g**2,'T2*gdppc2':10**2*g**2,
                               'P':Pall,'P*gdppc1':Pall*g,'P*gdppc2':Pall*g**2})

        X_pred = X_pred[dats[r].index]
        y_hat = X_pred.values @ dats[r]['Parameter'].values

        lc_ls = []
        lc_fs = []
        for ti in range(len(Pall))[1:-1]:
            gfrac = histP15[(histP15.gdppc>(g-0.5)) &
                            (histP15.gdppc<(g+0.5)) &
                            (histP15.Precipitation>Pall[ti-1]) &
                            (histP15.Precipitation<Pall[ti+1])
                            ]
            gfrac = gfrac[weightvar].sum()/histP15[weightvar].sum()*500
            col = plt.cm.copper((g-np.min(gdppcs))/7)
            lc_ls.append([(Pall[ti-1],y_hat[ti-1]),(Pall[ti],y_hat[ti])])
            lc_fs.append(gfrac)

        lc = LineCollection(lc_ls,lw=lc_fs,color=col,edgecolor=col,antialiased=False)
        ax.add_collection(lc)
        linesP.append(Line2D([0],[0],color=col,linewidth=3))
        labsP.append('GDPpc={}'.format(g))
    ax.set_ylabel('Climate-driven\nrural out-migration rate',fontsize=12)
    ax.set_xlabel('Precipitation',fontsize=12)
    ax.set_ylim(-0.2,0.27)
    ax.legend(linesP,labsP,loc=4)
    ax.set_xlim((0,30))

    plt.tight_layout()
    #plt.savefig('../figures/hist_parabolas_TP.png',dpi=400)
