import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.colors as mcolors
import pickle
import pdb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%matplotlib qt5

component = 'all' # all, T, or P. Really only T and P do anything

# Names of fixed-GDP and SSP-gdp runs
run_name     = 'gdptime_1990_quadT_linP'
run_name_ssp = 'sspgdp_'+run_name
run_name_fix = 'fixgdp_'+run_name

# ocean color
fc = [0.9,0.9,0.9]

# Whether to divide international migration numbers by country population (from shapefiles)
popnorm = False

# Custom colormaps
cmap_sio    = mcolors.LinearSegmentedColormap.from_list("sio", ['#7697B8','white','#F8B000'], N=256)
cmap_aus    = mcolors.LinearSegmentedColormap.from_list("sio", ['#34454F','white','#5C4729'], N=256)
cmap_tealor = mcolors.LinearSegmentedColormap.from_list("sio", ['#007A74','white','#F28500'], N=256)
cmap_tealor_r=mcolors.LinearSegmentedColormap.from_list("sio", ['#F28500','white','#007A74'], N=256)
cmap_indgol = mcolors.LinearSegmentedColormap.from_list("sio", ['#3C2F7D','white','#FFD700'], N=256)
cmap_blucor = mcolors.LinearSegmentedColormap.from_list("sio", ['#496D89','white','#FF5E5B'], N=256)

cmap_rate  = plt.cm.RdBu_r
cmap_inmig = cmap_tealor
cmap_oumig = cmap_tealor
#cmap_oumig = cmap_tealor_r

with open('../data/projections/{}/rep_full.pickle'.format(run_name_ssp), 'rb') as handle:
    projssp = pickle.load(handle)[0]
with open('../data/projections/{}/rep_term.pickle'.format(run_name_ssp), 'rb') as handle:
    projssp_term = pickle.load(handle)[0]
with open('../data/projections/{}/rep_full.pickle'.format(run_name_fix), 'rb') as handle:
    projfix = pickle.load(handle)[0]
with open('../data/projections/{}/rep_term.pickle'.format(run_name_fix), 'rb') as handle:
    projfix_term = pickle.load(handle)[0]

projssp_grid = pd.read_csv(
    '../data/projections/{}/{}_sspgdp_grid_projection.csv'.format(run_name_ssp,run_name_ssp),
    index_col=0
)
projfix_grid = pd.read_csv(
    '../data/projections/{}/{}_fixgdp_grid_projection.csv'.format(run_name_fix,run_name_fix),
    index_col=0
)

coefs = pd.read_csv('../data/projections/{}/reg/rep000.csv'.format(run_name_ssp),index_col=0)

term_mig_cols        = [c for c in projssp_term.columns if c.startswith('mig_rate')]
term_mig_cols        = [c for c in term_mig_cols if not c.endswith('_pop')]
term_mig_cols_inint  = [c for c in projssp_term.columns if (c.startswith('mig_num') & c.endswith('inint'))]
term_mig_cols_outint = [c for c in projssp_term.columns if (c.startswith('mig_num') & c.endswith('outint'))]

# Overwrite total mig rate with sum of all non-GDP-only terms
# No longer actually necessary because I exclude these terms from the projection entirely
exclude            = ['mig_rate_np.log(income)',      'mig_rate_np.log(income)*np.log(income)']
exclude_num_inint  = ['mig_num_np.log(income)_inint', 'mig_num_np.log(income)*np.log(income)_inint']
exclude_num_outint = ['mig_num_np.log(income)_outint','mig_num_np.log(income)*np.log(income)_outint']

if component in ['T','P']:
    exclude            += [c for c in term_mig_cols        if component not in c]
    exclude_num_inint  += [c for c in term_mig_cols_inint  if component not in c]
    exclude_num_outint += [c for c in term_mig_cols_outint if component not in c]

term_mig_cols_use        = [c for c in term_mig_cols        if c not in exclude]
term_mig_cols_use_inint  = [c for c in term_mig_cols_inint  if c not in exclude_num_inint]
term_mig_cols_use_outint = [c for c in term_mig_cols_outint if c not in exclude_num_outint]

projssp['mig_rate_tot'] = projssp_term[term_mig_cols_use].sum(1)
projfix['mig_rate_tot'] = projfix_term[term_mig_cols_use].sum(1)              # already equal unless component is T or P

projssp['mig_num_tot_inint'] = projssp_term[term_mig_cols_use_inint].sum(1)
projfix['mig_num_tot_inint'] = projfix_term[term_mig_cols_use_inint].sum(1)   # already equal unless component is T or P

projssp['mig_num_tot_outint'] = projssp_term[term_mig_cols_use_outint].sum(1)
projfix['mig_num_tot_outint'] = projfix_term[term_mig_cols_use_outint].sum(1) # already equal unless component is T or P

projssp['mig_num_tot_inint']  = projssp['mig_num_tot_inint']  / 1e6
projssp['mig_num_tot_outint'] = projssp['mig_num_tot_outint'] / 1e6
projfix['mig_num_tot_inint']  = projfix['mig_num_tot_inint']  / 1e6
projfix['mig_num_tot_outint'] = projfix['mig_num_tot_outint'] / 1e6

if popnorm:
    projssp['mig_num_tot_inint']  = projssp['mig_num_tot_inint']  / projssp['POP_EST'] * 1e6
    projssp['mig_num_tot_outint'] = projssp['mig_num_tot_outint'] / projssp['POP_EST'] * 1e6
    projfix['mig_num_tot_inint']  = projfix['mig_num_tot_inint']  / projfix['POP_EST'] * 1e6
    projfix['mig_num_tot_outint'] = projfix['mig_num_tot_outint'] / projfix['POP_EST'] * 1e6

projssp['mig_num_tot_netint'] = projssp['mig_num_tot_inint'] - projssp['mig_num_tot_outint']
projfix['mig_num_tot_netint'] = projfix['mig_num_tot_inint'] - projfix['mig_num_tot_outint']

# Migration rate colorbar limits
col_lims_tot  = {'full SSP':0.08,'fixed GDP':0.08}
col_lims_term = {'full SSP': {'T':0.10,'P':0.10,'Ttrend':0.10,'Ptrend':0.10,'Pvar':0.10,'Tvar':0.10},
                 'fixed GDP':{'T':0.10,'P':0.10,'Ttrend':0.10,'Ptrend':0.10,'Pvar':0.10,'Tvar':0.10}}

# International migration number colorbar limits
if popnorm:
    col_lims_int_tot  = {'full SSP':.01,'fixed GDP':.01}
else:
    col_lims_int_tot  = {'full SSP':1.5e6/1e6,'fixed GDP':1.5e6/1e6}

col_lims_int_term = {'full SSP': {'T':1e6/5,'P':1e6/5,'Ttrend':1e6/5,'Ptrend':1e6/5,
                                  'Pvar':1e6/5,'Tvar':1e6/5},
                     'fixed GDP':{'T':1e6/5,'P':1e6/5,'Ttrend':1e6/5,'Ptrend':1e6/5,
                                  'Pvar':1e6/5,'Tvar':1e6/5}}

col_scale = 1
col_lims_tot  = {k:v*col_scale for k,v in col_lims_tot.items()}
col_lims_term['full SSP'] = {k:v*col_scale for k,v in col_lims_term['full SSP'].items()}
col_lims_term['fixed GDP'] = {k:v*col_scale for k,v in col_lims_term['fixed GDP'].items()}

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Plot total migration rates
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
plt.figure(figsize=(11,8));
ax=plt.subplot(211,projection=ccrs.Robinson());
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_rate_tot',vmin=-col_lims_tot['full SSP'],vmax=col_lims_tot['full SSP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in outmigration rate 2050');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)

ax=plt.subplot(212,projection=ccrs.Robinson());
ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_rate_tot',vmin=-col_lims_tot['fixed GDP'],vmax=col_lims_tot['fixed GDP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in outmigration rate 2050, fixed income');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)
plt.tight_layout()

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Plot total international inmigration
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
plt.figure(figsize=(18,5));
ax=plt.subplot(231,projection=ccrs.Robinson());
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_num_tot_inint',vmin=-col_lims_int_tot['full SSP'],
                                   vmax= col_lims_int_tot['full SSP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in international inmigration 2050');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)

ax=plt.subplot(234,projection=ccrs.Robinson());
ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_num_tot_inint',vmin=-col_lims_int_tot['fixed GDP'],
                                   vmax= col_lims_int_tot['fixed GDP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in international inmigration 2050, fixed income');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)
plt.tight_layout()

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Plot international outmigrants
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
ax=plt.subplot(232,projection=ccrs.Robinson());
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_num_tot_outint',vmin=-col_lims_int_tot['full SSP'],
                             vmax= col_lims_int_tot['full SSP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in international outmigrants 2050');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)

ax=plt.subplot(235,projection=ccrs.Robinson());
ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_num_tot_outint',vmin=-col_lims_int_tot['fixed GDP'],
                             vmax= col_lims_int_tot['fixed GDP'],
                cmap='RdBu_r',ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, change in international outmigrants 2050, fixed income');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)
plt.tight_layout()

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Plot net international migrants
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

ax=plt.subplot(233,projection=ccrs.Robinson());
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_num_tot_netint',vmin=-col_lims_int_tot['full SSP'],
                             vmax= col_lims_int_tot['full SSP'],
                cmap=cmap_rate,ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, net change in international migrants 2050');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)

ax=plt.subplot(236,projection=ccrs.Robinson());
ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_num_tot_netint',vmin=-col_lims_int_tot['fixed GDP'],
                             vmax= col_lims_int_tot['fixed GDP'],
                cmap=cmap_rate,ax=ax,legend=True,transform=ccrs.PlateCarree());
plt.title('SSP2, net change in international migrants 2050, fixed income');
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax,transform=ccrs.PlateCarree());
ax.add_feature(cfeature.OCEAN, facecolor=fc)
plt.tight_layout()

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Plot 6-panel combined: rate, international in, international out
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

maxout = False
symlog = False

def get_extreme(dats,v):
    # dats is a list of objects to use for max/min
    out = 0
    for dat in dats:
        maxv = np.max(np.abs([dat[v].min(),dat[v].max()]))
        if maxv>out:
            out = maxv
    return [-out,out]

# Rates
fig = plt.figure(figsize=(13,5));
gs = gridspec.GridSpec(3,3, height_ratios=[1,1,0.4])
ax1  = fig.add_subplot(gs[0,0],projection=ccrs.Robinson())
ax2  = fig.add_subplot(gs[0,1],projection=ccrs.Robinson())
ax3  = fig.add_subplot(gs[0,2],projection=ccrs.Robinson())
ax4  = fig.add_subplot(gs[1,0],projection=ccrs.Robinson())
ax5  = fig.add_subplot(gs[1,1],projection=ccrs.Robinson())
ax6  = fig.add_subplot(gs[1,2],projection=ccrs.Robinson())
ax4c = fig.add_subplot(gs[2,0])
ax5c = fig.add_subplot(gs[2,1])
ax6c = fig.add_subplot(gs[2,2])

ax4c.axis('off')
ax5c.axis('off')
ax6c.axis('off')

# Gridded rates
# Convert ssp to xarray
newlat = np.linspace(-89.75,89.75,360)
ugridssp = projssp_grid.loc[(projssp_grid.year==2050)&(projssp_grid.ssp==2)]
ugridssp.loc[:,'mig_rate_tot_pop'] = ugridssp['mig_rate_tot']*ugridssp['pop_total_rural']
ugridssp_grp = ugridssp.groupby(['lat','lon'])[['mig_rate_tot_pop','pop_total_rural']].sum()
ugridssp_grp.loc[:,'mig_rate_wt'] = ugridssp_grp['mig_rate_tot_pop']/ugridssp_grp['pop_total_rural']
ugridssp_xr = ugridssp_grp.to_xarray()
ugridssp_xr = ugridssp_xr.reindex(lat=newlat,method=None)

# Convert fix to xarray
ugridfix = projfix_grid.loc[(projfix_grid.year==2050)&(projfix_grid.ssp==2)]
ugridfix.loc[:,'mig_rate_tot_pop'] = ugridfix['mig_rate_tot']*ugridfix['pop_total_rural']
ugridfix_grp = ugridfix.groupby(['lat','lon'])[['mig_rate_tot_pop','pop_total_rural']].sum()
ugridfix_grp.loc[:,'mig_rate_wt'] = ugridfix_grp['mig_rate_tot_pop']/ugridfix_grp['pop_total_rural']
ugridfix_xr = ugridfix_grp.to_xarray()
ugridfix_xr = ugridfix_xr.reindex(lat=newlat,method=None)

# Plot gridded later to use axis limit from geopandas axes

# International in-migrants
if maxout:
    ulims = get_extreme([projssp.loc[(projssp.year==2050)&(projssp.ssp==2)],
                         projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]],
                         'mig_num_tot_inint')
else:
    ulims = [-col_lims_int_tot['full SSP'],col_lims_int_tot['full SSP']]
if symlog:
    new = 10**np.ceil(np.log10(ulims[1]))
    ulims = [-new,new]
    
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
if symlog:
    # CHANGED
    ssp2_2050.plot('mig_num_tot_inint',
                    cmap=cmap_inmig,ax=ax6,legend=False,transform=ccrs.PlateCarree());
else:
    ssp2_2050.plot('mig_num_tot_inint',vmin=ulims[0],vmax=ulims[1],
                    cmap=cmap_inmig,ax=ax6,legend=False,transform=ccrs.PlateCarree());


ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax6,transform=ccrs.PlateCarree());
ax6.add_feature(cfeature.OCEAN, facecolor=fc)

ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
if symlog:
    norm7 = mpl.colors.SymLogNorm(vmin=ulims[0],vmax=ulims[1],linthresh=1)
    ssp2_2050.plot('mig_num_tot_inint',
    # CHANGED
                    cmap=cmap_inmig,ax=ax3,legend=True,transform=ccrs.PlateCarree(),
                    norm=norm7,
                    legend_kwds={'orientation':'horizontal','shrink':0.5});
else:
    ssp2_2050.plot('mig_num_tot_inint',vmin=ulims[0],vmax=ulims[1],
                    cmap=cmap_inmig,ax=ax3,legend=False,transform=ccrs.PlateCarree());
    norm7 = mpl.colors.Normalize(vmin=ulims[0],vmax=ulims[1])

ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax3,transform=ccrs.PlateCarree());
ax3.add_feature(cfeature.OCEAN, facecolor=fc)
sm5 = plt.cm.ScalarMappable(cmap=cmap_inmig,norm=norm7)

# International out-migrants
if maxout:
    ulims = get_extreme([projssp.loc[(projssp.year==2050)&(projssp.ssp==2)],
                         projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]],
                         'mig_num_tot_outint')
else:
    ulims = [-col_lims_int_tot['full SSP'],col_lims_int_tot['full SSP']]
if symlog:
    new = 10**np.ceil(np.log10(ulims[1]))
    ulims = [-new,new]
ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
if symlog:
    ssp2_2050.plot('mig_num_tot_outint',
                    cmap=cmap_oumig,ax=ax5,legend=False,transform=ccrs.PlateCarree(),
                    norm=mpl.colors.SymLogNorm(vmin=ulims[0],vmax=ulims[1],linthresh=1));

else:
    ssp2_2050.plot('mig_num_tot_outint',vmin=ulims[0],vmax=ulims[1],
                    cmap=cmap_oumig,ax=ax5,legend=False,transform=ccrs.PlateCarree());

ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax5,transform=ccrs.PlateCarree());
ax5.add_feature(cfeature.OCEAN, facecolor=fc)


ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
if symlog:
    norm8 = mpl.colors.SymLogNorm(vmin=ulims[0],vmax=ulims[1],linthresh=1)    
    ssp2_2050.plot('mig_num_tot_outint',vmin=ulims[0],vmax=ulims[1],
                    cmap=cmap_oumig,ax=ax2,legend=False,transform=ccrs.PlateCarree(),
                    norm=norm8)
else:
    ssp2_2050.plot('mig_num_tot_outint',vmin=ulims[0],vmax=ulims[1],
                    cmap=cmap_oumig,ax=ax2,legend=False,transform=ccrs.PlateCarree())
    norm8 = mpl.colors.Normalize(vmin=ulims[0],vmax=ulims[1])
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax2,transform=ccrs.PlateCarree());
ax2.add_feature(cfeature.OCEAN, facecolor=fc)
sm6 = plt.cm.ScalarMappable(cmap=cmap_oumig,norm=norm8)

# Plot gridded rates
rdbu = cmap_rate.copy()
rdbu.set_bad(color=fc)
ugridssp_xr_mig = ugridssp_xr['mig_rate_wt']
ugridssp_xr_mig.name = ''
ugridssp_xr_mig.plot(vmin=-col_lims_tot['full SSP'],vmax=col_lims_tot['full SSP'],
    cmap=rdbu,transform=ccrs.PlateCarree(),ax=ax4, add_colorbar=False)
ax4.set_xlim(ax2.get_xlim())
ax4.set_ylim(ax2.get_ylim())
ax4.add_feature(cfeature.BORDERS,linewidth=0.2,color='k')
ax4.add_feature(cfeature.COASTLINE,linewidth=0.2,color='k')


ugridfix_xr_mig = ugridfix_xr['mig_rate_wt']
ugridfix_xr_mig.name = ''
sm4 = ugridfix_xr_mig.plot(vmin=-col_lims_tot['full SSP'],vmax=col_lims_tot['full SSP'],
    cmap=rdbu,transform=ccrs.PlateCarree(),ax=ax1,add_colorbar=False)
ax1.set_xlim(ax5.get_xlim())
ax1.set_ylim(ax5.get_ylim())
ax1.add_feature(cfeature.BORDERS,linewidth=0.2,color='k')
ax1.add_feature(cfeature.COASTLINE,linewidth=0.2,color='k')

# Whole figure additions
plt.tight_layout()
ax4.annotate('Climate + Income',(-0.07,0.5),xycoords='axes fraction',
             ha='center',va='center',rotation=90,fontsize=13)
ax1.annotate('Climate only',(-0.07,0.5),xycoords='axes fraction',
             ha='center',va='center',rotation=90,fontsize=13)
plt.subplots_adjust(left=0.05, bottom=0.02, right=1-0.05, top=1-0.02, wspace=0.07, hspace=0.09)

# Add colorbars
labs = ['$\Delta$ climate-induced rural\noutmigration rate\n(Fraction / 5yr)',
        '$\Delta$ climate-induced\nInternational out-migrants\n(Mil / 5yr)',
        '$\Delta$ climate-induced\nInternational in-migrants\n(Mil / 5yr)']
cfmt = [None,None,None]
exts = ['both','both','both']
for cax,sm,lab,fmt,ext in zip([ax4c, ax5c, ax6c],[sm4,sm5,sm6],labs,cfmt,exts):
    axcb = cax.get_position()
    axcax = fig.add_axes([axcb.corners()[:,0].min()+0.022,axcb.corners()[:,1].min()+0.14, 
                           axcb.width*0.848, 0.030]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(sm, cax = axcax, orientation='horizontal',format=fmt,extend=ext)
    axcax.set_xlabel(lab,labelpad=0.07)
                 #extend='both',pad=0.1);

abc = ['A','B','C','D','E','F','G','H']
for axi,ax in enumerate([ax1,ax4,ax2,ax5,ax3,ax6]):
    ax.annotate(abc[axi],(0.02,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

#plt.savefig('../figures/{}__projection_maps.png'.format(run_name),dpi=500)

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   Add figure of relative proportions
acshares = pd.read_csv('../data/migration/abelcohen_shares.csv')
acshares = acshares.rename(columns={'ISO_A3':'adm0_country_id'})
# Rectify Romania's country code
acshares['adm0_country_id'] = acshares['adm0_country_id'].str.replace('ROU','ROM')
acshares['dest'] = acshares['dest'].str.replace('ROU','ROM')

base_outint = acshares.groupby('adm0_country_id')['da_pb_closed'].sum()
base_outint.name = 'base_outint'
base_inint  = acshares.groupby('dest')['da_pb_closed'].sum()
base_inint.name = 'base_inint'
base_inint.index.name = 'adm0_country_id'

projssp = projssp.merge(base_outint,on='adm0_country_id')
projssp = projssp.merge(base_inint, on='adm0_country_id')
projfix = projfix.merge(base_outint,on='adm0_country_id')
projfix = projfix.merge(base_inint, on='adm0_country_id')

# Calculate fraction (mig_num in millions, so need to undo that for this calculation)
projssp['mig_frac_inint']  = projssp['mig_num_tot_inint']  / projssp['base_inint'] *1e6
projssp['mig_frac_outint'] = projssp['mig_num_tot_outint'] / projssp['base_outint']*1e6
projfix['mig_frac_inint']  = projfix['mig_num_tot_inint']  / projfix['base_inint'] *1e6
projfix['mig_frac_outint'] = projfix['mig_num_tot_outint'] / projfix['base_outint']*1e6

#   #   #   #   #   #   #   #   #   #   #
#   Fractional Figure
fig = plt.figure(figsize=(8,5));
gs = gridspec.GridSpec(3,2, height_ratios=[1,1,0.4])
ax1  = fig.add_subplot(gs[0,0],projection=ccrs.Robinson())
ax2  = fig.add_subplot(gs[0,1],projection=ccrs.Robinson())
ax3  = fig.add_subplot(gs[1,0],projection=ccrs.Robinson())
ax4  = fig.add_subplot(gs[1,1],projection=ccrs.Robinson())
ax3c = fig.add_subplot(gs[2,0])
ax4c = fig.add_subplot(gs[2,1])

ax3c.axis('off')
ax4c.axis('off')

fraclim_in  = [-1,1]
fraclim_out = [-1,1]
ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_frac_outint',vmin=fraclim_out[0],vmax=fraclim_out[1],
                cmap=cmap_inmig,ax=ax1,legend=False,transform=ccrs.PlateCarree());
norm1 = mpl.colors.Normalize(vmin=fraclim_out[0],vmax=fraclim_out[1])
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax1,transform=ccrs.PlateCarree());
ax1.add_feature(cfeature.OCEAN, facecolor=fc)

ssp2_2050 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)]
ssp2_2050.plot('mig_frac_inint',vmin=fraclim_in[0],vmax=fraclim_in[1],
                cmap=cmap_inmig,ax=ax2,legend=False,transform=ccrs.PlateCarree());
norm2 = mpl.colors.Normalize(vmin=fraclim_in[0],vmax=fraclim_in[1])
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax2,transform=ccrs.PlateCarree());
ax2.add_feature(cfeature.OCEAN, facecolor=fc)

ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_frac_outint',vmin=fraclim_out[0],vmax=fraclim_out[1],
               cmap=cmap_inmig,ax=ax3,legend=False,transform=ccrs.PlateCarree());
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax3,transform=ccrs.PlateCarree());
ax3.add_feature(cfeature.OCEAN, facecolor=fc)
sm3 = plt.cm.ScalarMappable(cmap=cmap_inmig,norm=norm1)

ssp2_2050 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)]
ssp2_2050.plot('mig_frac_inint',vmin=fraclim_in[0],vmax=fraclim_in[1],
               cmap=cmap_inmig,ax=ax4,legend=False,transform=ccrs.PlateCarree());
ssp2_2050.boundary.plot(color='k',lw=0.1,ax=ax4,transform=ccrs.PlateCarree());
ax4.add_feature(cfeature.OCEAN, facecolor=fc)
sm4 = plt.cm.ScalarMappable(cmap=cmap_inmig,norm=norm2)

plt.tight_layout()
ax3.annotate('Climate + Income',(-0.07,0.5),xycoords='axes fraction',
             ha='center',va='center',rotation=90,fontsize=13)
ax1.annotate('Climate only',(-0.07,0.5),xycoords='axes fraction',
             ha='center',va='center',rotation=90,fontsize=13)
plt.subplots_adjust(left=0.05, bottom=0.02, right=1-0.05, top=1-0.02, wspace=0.07, hspace=0.09)

labs = ['Fractional $\Delta$ climate-induced\nInternational out-migrants\n',
        'Fractional $\Delta$ climate-induced\nInternational in-migrants\n']
cfmt = [None,None]
exts = ['both','both','both']
for cax,sm,lab,fmt,ext in zip([ax3c, ax4c],[sm3,sm4],labs,cfmt,exts):
    axcb = cax.get_position()
    axcax = fig.add_axes([axcb.corners()[:,0].min()+0.022,axcb.corners()[:,1].min()+0.14, 
                           axcb.width*0.9, 0.030]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(sm, cax = axcax, orientation='horizontal',format=fmt,extend=ext)
    axcax.set_xlabel(lab,labelpad=0.07)
                 #extend='both',pad=0.1);

abc = ['A','B','C','D','E','F','G','H']
for axi,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.annotate(abc[axi],(0.02,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   Get numbers for largest changes
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

printcols = ['adm0_country_id','mig_num_tot_outint','mig_num_tot_inint']
fix50 = projfix.loc[(projfix.year==2050)&(projfix.ssp==2)][printcols]
ssp50 = projssp.loc[(projssp.year==2050)&(projssp.ssp==2)][printcols]

fix50 = fix50.set_index('adm0_country_id')
ssp50 = ssp50.set_index('adm0_country_id')

selected_countries = ['ZAF','NGA','KEN','ETH','CHN','JPN','AUS','VNM','IND','SAU',
                      'GRC','RUS','DEU','ESP','PAN','MEX','USA','ARG','CHL','BRA']
printssp = ssp50.loc[selected_countries]*1e6
printfix = fix50.loc[selected_countries]*1e6
printssp = printssp.rename(columns={
    'mig_num_tot_outint':'SSP income, out-migration',
    'mig_num_tot_inint': 'SSP income, in-migration'})
printfix = printfix.rename(columns={
    'mig_num_tot_outint':'Fix income, out-migration',
    'mig_num_tot_inint': 'Fix income, in-migration'})
printmig = printssp.join(printfix).sort_values('SSP income, out-migration',
     ascending=False)
printmig = printmig.astype(int)
printmig.index.name = 'country'
r"\begin{tabular}{c}SSP income,\\ out-migration\end{tabular}"
printmig.columns = [r'\begin{{tabular}}{{c}}{}\\{}\end{{tabular}}'.format(*col.split(',')) for col in printmig.columns]
print(printmig.to_latex(index=True, escape=False))


