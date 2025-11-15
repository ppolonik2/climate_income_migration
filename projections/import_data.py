"""
This script reads and formats the input data for the projections
  population, rural cells, climate, income
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import sys
import namelist_run as nl
sys.path.append('{}/utils/'.format(nl.root_dir))
import utilfuncs as uf
import pdb

def fullpath(pathlist):
    """
    Shortcut to read files
    """
    fp = os.path.join(nl.dat_dir,*pathlist)
    return fp

def read_xr(pathlist):
    fp = fullpath(pathlist)
    dat = uf.reorient_netCDF(xr.open_dataset(fp))
    return dat

## Import SSP population data csv and store in a dictionary
def load_pop():
    pop  = pd.read_csv(fullpath(['population','SSP','processed',
                      'popc_ssp_rural_halfdeg_duplicate.csv']),
                      index_col=0)
    pop  = pop[pop['ssp'].isin([1,2,3,5])]
    pop  = pop.rename(columns={'time':'year','udelID':'cell_id'})
    pop2 = pop[['cell_id','adm1_state_id','ssp','year','popc']]

    # Interpolate to each 5-year from decade
    old_years   = np.sort(pop2['year'].unique())
    new_years   = range(old_years[0],old_years[-1]+1,5)
    pop_expand  = pop2.set_index(['cell_id', 'adm1_state_id', 'ssp', 'year']).unstack().T
    pop_expand  = pop_expand.loc['popc'].reindex(new_years)
    pop_expand  = pop_expand.interpolate(method='linear',axis=0)
    pop_expand  = pop_expand.unstack()

    pop_new = pop_expand.reset_index().rename(columns={0:'popc'})
    map_var = ['cell_id','adm1_state_id','adm0_country_id','lat','lon','area','cellfrac']
    pop_map = pop[map_var].drop_duplicates()
    pop_new = pop_new.merge(pop_map,on=['cell_id','adm1_state_id'])
    pop_new['popd'] = pop_new['popc']/(pop_new['area']*pop_new['cellfrac'])

    return pop_new

## Import csv that specifies rural cells (1 = in_sample, 0 = out_sample)
# This was created by Ida
def load_rural_cells():
    rural_cells = pd.read_csv(fullpath(['population','other','rural_cells.csv']))
    rural_cells = rural_cells.rename(columns={'latitude':'lat', 'longitude':'lon'})
    return rural_cells

def load_clim(meanopt=False):
    ## Import climate input files
    ssps = {1:'ssp126',2:'ssp245',3:'ssp370',5:'ssp585'}
    not_models = ['Mean','10th_percentile','90th_percentile']
    clim = []
    meanfile = fullpath(['climate','climate_metrics_modmean_haldfdeg_5yr.nc'])
    if os.path.exists(meanfile) & meanopt:
        clim = uf.reorient_netCDF(xr.open_dataset(meanfile))
    else:
        for ssp,SSP in ssps.items():
            climate = read_xr(['climate','climate_metrics_{}_halfdeg_5yr.nc'.format(SSP)])
            climate = climate.sel(model=[m for m in climate.model.values if m not in not_models])
            # Fill line at zero lon
            zeroslice = climate.sel(lon=slice(-2,2)).interpolate_na(dim='lon')
            for var in climate.data_vars:
                climate[var].loc[{'lat':zeroslice.lat,'lon':zeroslice.lon}] = zeroslice[var]

            if meanopt:
                climate = climate.mean('model')
                climate = climate.assign_coords({'model':['mean']})
            climate = climate.assign_coords({'ssp':[ssp]})
            clim.append(climate)
        clim = xr.concat(clim,'ssp')
        clim = clim.assign_coords({'time':clim.time.astype(int)})

        if meanopt:
            clim.to_netcdf(meanfile)

    # Add mean and anomaly
    Tmean = clim.sel(time=nl.baseline_yr)['Temperature'].broadcast_like(clim['Temperature'])
    clim['Tmean'] = Tmean
    clim['Tanom'] = clim['Temperature'] - clim['Tmean'] # Overwrite anomaly with this definition

    # Add lags
    clim['Tm1']  = clim['Temperature'].shift(time=1)
    clim['Pm1']  = clim['Precipitation'].shift(time=1)
    clim['T2m1'] = clim['Tm1']**2
    clim['P2m1'] = clim['Pm1']**2

    return clim


def load_gdp_oldgrid():
    gdp_ssp = read_xr(['gdp','processed','gdp_pop_ssps.nc'])
    # Interpolate every year and take 5-year mean
    new_years = np.arange(gdp_ssp.year[0],gdp_ssp.year[-1]+1)
    gdp_ssp_interp = gdp_ssp.interp(year=new_years)
    gdp_ssp_interp['year'] = pd.to_datetime(gdp_ssp_interp['year'].values,format='%Y')
    gdp_ssp = gdp_ssp_interp.resample(year='5AS').mean()
    # Drop the last 5-year interval (nans)
    gdp_ssp = gdp_ssp.isel(year=slice(None,-1))
    # Shift interval by 5 years to match JRC naming convention
    gdp_ssp.coords['year'] = gdp_ssp.year.dt.year+5
    # Convert year from float to str
    gdp_ssp.coords['year'] = gdp_ssp.coords['year']
    # Rename year to time
    gdp_ssp = gdp_ssp.rename({'year':'time','gdppc':'income'})
    # Convert income from $ 2005 to $ 2011 to match Kummu
    gdp_ssp['income'] = gdp_ssp['income']*1.15

    gdp_hist = xr.open_dataset('/home/ppolonik/../j4wan/Migration/Pop_mig/Pop_mig_data/gdp_05deg_gridmean.nc')
    gdp_hist = gdp_hist.assign_coords({'time':[2015]}).expand_dims({'ssp':gdp_ssp.ssp})
    gdp_hist = gdp_hist.rename({'gridmean_income':'income'})
    gdp_hist = gdp_hist.assign_coords({'lat':gdp_hist.lat.values[::-1]})
    gdp_ssp  = xr.concat([gdp_hist,gdp_ssp],'time')

    return gdp_ssp

def load_gdp_pop():
    fp = os.path.join(nl.dat_dir,'gdp','processed','rural_gdppc_05deg.csv')
    dat = pd.read_csv(fp,index_col=0)
    dat.loc[dat['gdppc']<0] = np.nan

    # Include fixed GDP baseline
    basegdp = dat.loc[dat.year==nl.baseline_yr]
    basegdp = basegdp.rename(columns={'gdppc':'base_gdppc'})
    dat = dat.merge(basegdp[['lat','lon','adm1_state_id','ssp','base_gdppc']],
                    on  = ['lat','lon','adm1_state_id','ssp'],
                    how = 'left')
    return dat

def load_master_inputs(meanclim=False):
    clim = load_clim(meanopt=meanclim)
    clim = clim.rename({'time':'year'})

    gdp_pop = load_gdp_pop()
    allyrs  = gdp_pop.year.unique()
    allyrs  = allyrs[~np.isnan(allyrs)]
    climdf  = clim.sel(year=allyrs).to_dataframe()
    
    # Expand gdp_pop to have a model dimension to allow for merge on models
    allmodels = pd.DataFrame(climdf.index.get_level_values('model')).drop_duplicates()
    gdp_pop['tmpkey'] = 1
    allmodels['tmpkey']  = 1
    gdp_pop = gdp_pop.merge(allmodels,on='tmpkey').drop(columns='tmpkey')

    # Merge and subset to valid data
    master_inputs = gdp_pop.merge(climdf,on=['lat','lon','year','ssp','model',],how='left')
    master_inputs = master_inputs.loc[(master_inputs.year>=nl.baseline_yr)&
                                      (master_inputs.year<=nl.end_yr)]

    master_inputs = master_inputs.loc[~np.isnan(master_inputs.Temperature) & 
                                      ~np.isnan(master_inputs.base_gdppc)  &
                                      ~np.isnan(master_inputs.gdppc)
                                      ]
    if nl.const_gdp:
        master_inputs['gdppc'] = master_inputs['base_gdppc']

    # When selected,load in robustness test results and merge with master_inputs in order to cut out non-passing cells
    if nl.pass_test_only:
        pass_test = pd.read_csv(fullpath(['regres','robustness_T_gdppc1_mean.csv']),index_col=0)
        pass_test['pass'] = np.abs(pass_test['SE_error']) < 1.959963984540054
        pass_test = pass_test.rename(columns={'lat_mean':'lat','lon_mean':'lon','admin_id':'adm1_state_id'})
        master_index = master_inputs.index
        master_inputs = master_inputs.merge(pass_test[['lat','lon','adm1_state_id','pass']],how='left')
        master_inputs['pass'] = master_inputs['pass'].fillna(False)
        master_inputs.index = master_index
        master_inputs = master_inputs.loc[master_inputs['pass']]

    return master_inputs

def load_countries():
    # Read in country geometry file
    countries = gpd.read_file(fullpath(['political','ne_50m_admin_0_countries.shp']))
    # Norway (ISO_N3=-99, need to manually add to dataframe)
    countries.loc[88, 'ISO_A3']='NOR'
    countries.loc[160,'ISO_A3']='FRA'
    countries = countries.rename(columns={'ISO_A3':'adm0_country_id'})
    # Remove -99 adm0_country_id (5 rows)
    countries = countries.drop(countries[countries['adm0_country_id']=='-99'].index)
    countries['adm0_country_id'] = countries.adm0_country_id.astype(str)
    return countries

def load_ac_shares():
    filepath = fullpath(['migration','abelcohen_shares.csv'])
    ac_shares = pd.read_csv(filepath)
    ac_shares = ac_shares.rename(columns={'ISO_A3':'adm0_country_id'})
    # Using constant beta defined in namelist, so dropping column here to avoid confusion
    ac_shares = ac_shares.drop(columns='estimated_beta_ipums')

    # Rectify Romania's country code
    ac_shares['adm0_country_id'] = ac_shares['adm0_country_id'].str.replace('ROU','ROM')
    ac_shares['dest'] = ac_shares['dest'].str.replace('ROU','ROM')

    return ac_shares
     

