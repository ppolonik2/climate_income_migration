import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import seaborn as sns
import pdb
import sys
sys.path.append('../utils/')
import utilfuncs as uf
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
#%matplotlib qt5

add_contours = True
add_arrows   = True
ccodes       = ['USA','CHN','IND','DEU','ZAF','ETH']
ssp          = 2
start_yr     = 2015
end_yr       = 2050
#run_name     = 'gdptime_1990_quadT_linP'
run_name     = 'gdptime_1990_quadT_linP_urban'

bonecmap = mcolors.LinearSegmentedColormap.from_list("gray", [[0]*3,[0.9]*3], N=256)


# Read data
run_name_ssp = 'sspgdp_'+run_name
run_name_fix = 'fixgdp_'+run_name
outdir = '../data/projections/{}'.format(run_name_ssp)
pyreg = pd.read_csv('{}/reg/rep000.csv'.format(outdir),index_col=0)

# This needs to be saved out manually
dat = pd.read_csv('{}/{}_full_05deg.csv'.format(outdir,run_name_ssp),index_col=0)

# Historical data going into regressions. Also saved out manually
regdat = pd.read_csv('../data/projections/sspgdp_gdptime_1990_quadT_linP/regres_dat.csv',index_col=0)
def get_slope(row):
    slope = np.polyfit(row.index, row.values, 1)[0]
    return slope
Tpiv = regdat.reset_index().pivot(index='entry_id',columns=['year'],values='T')
Tslope = Tpiv.apply(get_slope, axis=1)
Tslope.name = 'Tslope'
Ppiv = regdat.reset_index().pivot(index='entry_id',columns=['year'],values='P')
Pslope = Ppiv.apply(get_slope, axis=1)
Pslope.name = 'Pslope'
regdat = regdat.merge(Tslope,on='entry_id')
regdat = regdat.merge(Pslope,on='entry_id')

def weighted_violin(ax,x,vals,wts,width=0.4):
    # Weighted KDE
    kde = gaussian_kde(vals, weights=wts)
    y = np.linspace(vals.min(), vals.max(), 500)
    density = kde(y)

    # Normalize width
    density = density / density.max() * width

    # Mirror the density to make a violin shape
    #ax.fill_betweenx(y, x - density, x + density, label='',color='#A5927B',alpha=0.8)
    ax.fill_betweenx(y, x - density, x + density, label='',color='#3182BD',alpha=0.5)

def weighted_violin_y(ax,y,vals,wts,width=0.4):
    # Weighted KDE
    kde = gaussian_kde(vals, weights=wts)
    x = np.linspace(vals.min(), vals.max(), 500)
    density = kde(x)

    # Normalize width
    density = density / density.max() * width

    # Mirror the density to make a violin shape
    #ax.fill_between(x, y - density, y + density, label='',color='#A5927B',alpha=0.8)
    ax.fill_between(x, y - density, y + density, label='',color='#3182BD',alpha=0.5)

# Make Contour
Ncont   = 200
Ts      = np.linspace(-5,35,Ncont)
Ps      = np.linspace(0,30,Ncont)
Ttrends = np.linspace(-0.5,1,Ncont)
gdppcs  = np.linspace(5.5,12.5,Ncont)
xs0 = {'T':Ts,'P':Ps,'Ttrend':Ttrends,'gdppc':gdppcs}
if 'urban' in run_name:
    clims = {'T':np.array([-0.1,0.30])-0.12,'P':np.array([-0.35,0.05]),'Ttrend':[-0.1,0.20]}
else:
    clims = {'T':[-0.1,0.30],'P':[-0.35,0.05],'Ttrend':[-0.1,0.20]}
xs  = xs0.copy()
for v,x in xs0.items():
    xs[v+'2'] = x**2
xs['gdppc1'] = xs['gdppc']

mainvar = pd.Series(
            pyreg.index.str.replace('1','').str.replace('2','').str.replace(' ','').str.split('*')
          )
mainvar2 = mainvar.explode()
mainvar2 = mainvar2.str.strip()
vs = mainvar2.explode().unique()
vs = [v for v in vs if v not in ['constant','gdppc']]
vmap     = {'T':'Temperature',          'P':'Precipitation',     'Ttrend':'Ttrend'}
vmap_lab = {'T':'Temperature [\u00b0C]','P':'Precipitation [dm]','Ttrend':'Ttrend'}
#scols = {1:'lightblue',2:'darkblue',3:'goldenrod',5:'firebrick'}
scols = {1:'#F5A560',2:'#F57F73',3:'#ED1E24',5:'#8A171A'}
fig = plt.figure(figsize=(7*len(vs),7.7))
gs = gridspec.GridSpec(2, 3*len(vs), width_ratios=[0.3,1,0.04,0.04,1,0.3], height_ratios=[0.6,1])
axTop    = [fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,4])]
axTopCol = [fig.add_subplot(gs[1,2]),fig.add_subplot(gs[1,3])]
axSide   = [fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,5])]
axPara   = [fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,4])]
for vi,v in enumerate(vs):
    print(v)
    p = pyreg.loc[(mainvar.str[0]==v).values,'Parameter']
    p.index = p.index.str.replace(' ','')
    Y = np.zeros((Ncont,Ncont))
    VARS = {}
    for eq in p.index.values:
        print(eq)
        es = eq.split('*')
        GDPPCS, XS = np.meshgrid(gdppcs,xs[eq.split('*')[0]])
        VARS[eq.split('*')[0]] = XS
        VARS['gdppc1'] = GDPPCS
        VARS['gdppc2'] = GDPPCS**2
        y = np.ones((Ncont,Ncont))
        for e in eq.split('*'):
            y *= VARS[e]
        print(p.loc[eq])
        Y += p.loc[eq] * y
    levels = np.linspace(clims[v][0], clims[v][1], 25+1)
    cf = axTop[vi].contourf(VARS['gdppc1'],VARS[v],Y,levels=levels,cmap=bonecmap)
    cbar = plt.colorbar(cf,cax=axTopCol[vi])
    if vi==1:
        axTopCol[vi].yaxis.set_ticks_position('left')
        axTop[vi].yaxis.set_ticks_position('right')
        axSide[vi].yaxis.set_label_position('left')

    if len(ccodes)>0:
        for ccode in ccodes:
            # To convert back to single arrow: 
            #  Remove assp loop and un-indent everything in that loop
            #  Change color=scols[assp] to color='k' and lw=1.5 to lw=3
            #for assp in [1,2,3,5]:
            for assp in [2]:
                datc = dat.loc[(dat['adm0_country_id']==ccode)&(dat['ssp']==assp)].dropna()
                if add_arrows:
                    vc = datc.pivot(index=['lat','lon','adm1_state_id'],
                                    columns='year',values=vmap[v]).dropna()
                    gc = datc.pivot(index=['lat','lon','adm1_state_id'],
                                    columns='year',values='gdppc').dropna()
                    pc = datc.pivot(index=['lat','lon','adm1_state_id'],
                                    columns='year',values='pop_total_rural').dropna()
                    vc = vc[[start_yr,end_yr]]
                    gc = gc[[start_yr,end_yr]]
                    pc = pc[[start_yr,end_yr]]
        
                    pop = pc.mean(1)
        
                    vw1 = np.sum(vc.iloc[:,0]*pop) / np.sum(pop) 
                    vw2 = np.sum(vc.iloc[:,1]*pop) / np.sum(pop) 
                    gw1 = np.sum(gc.iloc[:,0]*pop) / np.sum(pop) 
                    gw2 = np.sum(gc.iloc[:,1]*pop) / np.sum(pop) 
        
                    axTop[vi].annotate('',          # No text
                        xy=(gw2,vw2),               # End point
                        xytext=(gw1,vw1),           # Start point
                        #arrowprops=dict(arrowstyle='->', color='k', lw=3)
                        arrowprops=dict(arrowstyle='->', color=scols[assp], lw=1.5)
                    )

                    if assp==ssp:
                        if vi==0:
                            axTop[vi].text((gw1+gw2)/2,(vw1+vw2)/2-0.8,ccode,ha='center',
                                     va='top',fontsize=13,color='k')
                        else:
                            if ccode == 'ETH':
                                axTop[vi].text((gw1+gw2)/2-0.4,(vw1+vw2)/2-0.4,ccode,
                                         ha='center',va='top',fontsize=13,color='k')
                            else:
                                axTop[vi].text((gw1+gw2)/2,(vw1+vw2)/2-0.4,ccode,
                                         ha='center',va='top',fontsize=13,color='k')
    
    if (add_contours):
        # Add 2d hist contour if selected
        datu = dat.loc[(dat.year==2015)|(dat.year==2050)]
        datu = datu.loc[datu.ssp==ssp]
        f = 0.3
        datu = datu.loc[dat.ssp==ssp]
        datu = datu.sample(frac=f)
        sns.kdeplot(data=datu,x='gdppc',y=vmap[v],hue='year',palette=['#6BAED6','#08519C'],
                    weights=datu['pop_total_rural'],levels=[0.05,0.275,0.5,0.775,0.95],
                    linewidths=1,alphas=0.5,legend=False,ax=axTop[vi])
                                                    
    axTop[vi].set_ylim([xs0[v].min(),xs0[v].max()])
    axTop[vi].set_xlim([xs0['gdppc'].min(),xs0['gdppc'].max()])
    axTop[vi].set_xlabel('log(GDPpc)',fontsize=13)

    # Add violin plots on the sides
    lims = np.linspace(xs0[v].min(),xs0[v].max(),10)
    for i, (limlow, limhigh) in enumerate(zip(lims[:-1],lims[1:])):
        subset  = regdat.loc[(regdat[v]>limlow) & (regdat[v]<limhigh)]
        subset  = subset[[v+'slope','pop_base']].dropna()
        quant   = uf.weighted_quantile(subset[v+'slope'], [0.01,0.99], 
                                       sample_weight=subset['pop_base'])
        subset  = subset.loc[(subset[v+'slope']>quant[0]) & (subset[v+'slope']<quant[1])]
        values  = subset[v+'slope']
        weights = subset['pop_base']
        weighted_violin_y(axSide[vi],(limlow+limhigh)/2,values,weights,width=(limhigh-limlow)*0.35)

    axSide[vi].set_ylim(axTop[vi].get_ylim())
    axSide[vi].set_ylabel(vmap_lab[v],fontsize=13)
    axSide[vi].set_xlabel('Past trend in {}'.format(v),fontsize=13)
    axSide[vi].plot([0,0],axSide[vi].get_ylim(),'--k')
    if vi==1:
        axSide[vi].set_ylabel(vmap_lab[v],fontsize=13,rotation=270,labelpad=18)
        axSide[vi].yaxis.set_ticks_position('right')
        axSide[vi].yaxis.set_label_position('right')
    else:
        axSide[vi].set_ylabel(vmap_lab[v],fontsize=13)

# Add parabolas
param  = pyreg['Parameter']
Tmeans = [5, 10, 15, 20, 25, 30]
Tas    = np.linspace(-2,2)
Tall   = np.linspace(0,35,200)
Tcol   = [plt.cm.coolwarm(Tm/np.max(Tmeans)) for Tm in Tmeans]
Tcol[2] = plt.cm.coolwarm(Tmeans[2]/np.max(Tmeans)-0.05)

Pmeans =  [2, 4, 8, 16, 32]
Pbins  = [1,3, 6, 12, 24, 40]
Pas    = np.linspace(-4,4)
Pall   = np.linspace(0,35)
Pcol   = [plt.cm.BrBG(Pi/len(Pmeans)) for Pi in range(len(Pmeans))]
Pcol[2] = plt.cm.BrBG(2/len(Pmeans)-0.1)
Pcol[3] = plt.cm.BrBG(3/len(Pmeans)+0.1)
gdps   = [6,7,8,9,10,11,12]
gdpall = np.linspace(5,13,200)

curvesigT = {}
curvesigP = {}
for Tm in Tmeans:
    curvesigT[Tm] = {}
    curvemean = (param.loc['T']*Tm +
                 param.loc['T*gdppc1']* Tm*gdpall +
                 param.loc['T*gdppc2']* Tm*gdpall*gdpall
                 )
    if 'quadT' in run_name:
        curvemean = (curvemean +
                     param.loc['T2']*Tm*Tm +
                     param.loc['T2*gdppc1']*Tm*Tm*gdpall +
                     param.loc['T2*gdppc2']*Tm*Tm*gdpall*gdpall
                     )
    curvesigT[Tm] = curvemean

for Pm in Pmeans:
    curvesigP[Pm] = {}
    curvemean = (param.loc['P']*Pm +
                 param.loc['P*gdppc1']* Pm*gdpall +
                 param.loc['P*gdppc2']* Pm*gdpall*gdpall
                 )
    curvesigP[Pm] = curvemean

# Prep for histograms
histdat = pd.read_csv('../data/projections/{}/{}_full_05deg.csv'.format(run_name_ssp,run_name_ssp),index_col=0)
histdat = histdat.loc[(histdat.year==2015)|(histdat.year==2050)&(histdat.ssp==2)]
histdat['gdppc'] = histdat['gdppc'].clip(5,13)

histT15 = histdat.loc[histdat.year==2015][['Temperature','gdppc','pop_total_rural']]
histP15 = histdat.loc[histdat.year==2015][['Precipitation','gdppc','pop_total_rural']]
linesT = []
labsT = []

for Ti,Tm in enumerate(Tmeans):
    lc_ls = []
    lc_fs = []
    for si in range(len(gdpall))[1:-1]:
        Tfrac = histT15[(histT15.Temperature>(Tm-2.5)) &
                        (histT15.Temperature<(Tm+2.5)) &
                        (histT15.gdppc>gdpall[si-1]) &
                        (histT15.gdppc<gdpall[si+1])
                        ]
        Tfrac = Tfrac['pop_total_rural'].sum()/histT15['pop_total_rural'].sum()*500
        col = Tcol[Ti]
        lc_ls.append([(gdpall[si-1],curvesigT[Tm][si-1]),(gdpall[si],curvesigT[Tm][si])])
        lc_fs.append(Tfrac)

    lc = LineCollection(lc_ls,lw=lc_fs,color=col,edgecolor=col,antialiased=False)
    axPara[0].add_collection(lc)
    linesT.append(Line2D([0],[0],color=col,linewidth=3))
    labsT.append('T={}'.format(Tm))

axPara[0].set_ylabel('Climate-driven\nrural out-migration rate',fontsize=12)
if 'urban' in run_name:
    axPara[0].set_ylim(-0.2,0.17)
    axPara[0].legend(linesT,labsT,loc=2)
else:
    axPara[0].set_ylim(-0.1,0.27)
    axPara[0].legend(linesT,labsT,loc=3)
axPara[0].set_xlim(axTop[0].get_xlim())

linesP = []
labsP = []
for Pi,Pm in enumerate(Pmeans):
    lc_ls = []
    lc_fs = []
    for si in range(len(gdpall))[1:-1]:
        Pfrac = histP15[(histP15.Precipitation>(Pbins[Pi])) &
                        (histP15.Precipitation<(Pbins[Pi+1])) &
                        (histP15.gdppc>gdpall[si-1]) &
                        (histP15.gdppc<gdpall[si+1])
                        ]
        Pfrac = Pfrac['pop_total_rural'].sum()/histP15['pop_total_rural'].sum()*500
        col = Pcol[Pi]
        lc_ls.append([(gdpall[si-1],curvesigP[Pm][si-1]),(gdpall[si],curvesigP[Pm][si])])
        lc_fs.append(Pfrac)

    lc = LineCollection(lc_ls,lw=lc_fs,color=col,edgecolor=col,antialiased=False)
    axPara[1].add_collection(lc)
    linesP.append(Line2D([0],[0],color=col,linewidth=3))
    labsP.append('P={}'.format(Pm))
axPara[1].set_ylabel('Climate-driven\nrural out-migration rate',fontsize=12)
if 'urban' in run_name:
    axPara[1].legend(linesP,labsP,loc=2)
    axPara[1].set_ylim(-0.01,0.09)
else:
    axPara[1].legend(linesP,labsP,loc=3)
    axPara[1].set_ylim(-0.08,0.02)
axPara[1].set_xlim(axTop[1].get_xlim())


plt.subplots_adjust(left=0.05, bottom=0.08, right=1-0.05, top=1-0.04, wspace=0.2, hspace=0.18)
axTopCol[1].yaxis.set_label_position('left')
axTop[0].set_ylabel('log(GDP$_{pc}$)',fontsize=13)
axTop[1].set_ylabel('',fontsize=13)
axTop[0].text(5.8,-3.5,'Climate-driven\nrural out-migration rate',fontsize=13,color=[0.8]*3)
axSide[0].set_xlim([-0.1,0.1])
axSide[1].set_xlim([-0.25,0.25])

# Manually move the colorbars closer
cpos0 = axTopCol[0].get_position()
axTopCol[0].set_position([cpos0.x0 - 0.022, cpos0.y0, cpos0.width, cpos0.height])
cpos1 = axTopCol[1].get_position()
axTopCol[1].set_position([cpos1.x0 + 0.022, cpos1.y0, cpos1.width, cpos1.height])

# Add letters
abc = ['A','B','C','D','E','F']
for axi,ax in enumerate([axPara[0],axPara[1],axSide[0],axTop[0],axTop[1],axSide[1]]):
    ax.annotate(abc[axi],(0,1.00),xycoords='axes fraction',ha='left',va='bottom',fontsize=14)

#plt.savefig('../figures/parabolas_contours.png',dpi=500)
if 'urban' in run_name:
    plt.savefig('../figures/parabolas_contours_ssp2arrow_urban.png',dpi=500)
else:
    plt.savefig('../figures/parabolas_contours_ssp2arrow.png',dpi=500)

