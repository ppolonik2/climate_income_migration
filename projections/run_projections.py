import numpy as np
import import_data as impdat
import pandas as pd
import geopandas as gpd
import pickle
import copy
import re
import pdb
import matplotlib.pyplot as plt
import importlib
import os
import sys
sys.path.append('../../../utils/')
import stataTable as st
import namelist_run as nl

#   #   #   #   #   #   #   #   #   #
#   Read data
#   #   #   #   #   #   #   #   #   #
print('reading data...')
inp       = impdat.load_master_inputs(meanclim=nl.mean_opt)
countries = impdat.load_countries()
ac_shares = impdat.load_ac_shares()
print('finished reading data.')

#   #   #   #   #   #   #   #   #   #
#   Read in all the desired stata data to project
#   #   #   #   #   #   #   #   #   #

def short(term):
    if ('*' in term) and (term.strip().startswith('c')):
        return '*'.join(term.split('*')[1:]).strip()
    else:
        return term

def read_stata(blocks=nl.block_num, f=nl.stata_reg, individual=False):
    #  
    statas = {}
    statas = {}
    for block in blocks:
        statas[block] = {}
        if nl.reg_type=='python':
            block_placeholder = 1
        else:
            block_placeholder = block
        stata = st.stataTable(f,sheet=[],column=block_placeholder, 
                              log_inc=nl.flag_log_inc,min_sig=nl.minimum_significance)
        stata.load()

        # Overwrite with python results if selected
        if nl.reg_type=='python':
            python_reg = os.path.join(nl.out_dir,'reg')
            pyreg = pd.read_csv('{}/rep{}.csv'.format(python_reg,str(block).zfill(3)),
                                index_col=0)
            if nl.minimum_significance==2:
                pyreg = pyreg.loc[pyreg['P-value']<0.05]

            # Deal with squared variables
            pyreg.index = pyreg.index.str.strip()
            newi = []
            exclude = ['np.log(income)','np.log(income)*np.log(income)','constant']
            for i in pyreg.index:
                tms = i.split('*')
                newtms = '*'.join(['{}*{}'.format(tm[:-1],tm[:-1]) if tm.strip().endswith('2') 
                                    else tm for tm in tms])
                newtms = newtms.replace('gdppc1','np.log(income)')
                newtms = newtms.replace('gdppc','np.log(income)')
                newi.append(newtms)
            pyreg.index = newi

            # Remove terms not to include in projection
            for e in exclude:
                if e in pyreg.index:
                    pyreg = pyreg.drop(e)

            eqn = '+'.join(['c{}*{}'.format(i,ind) for i,ind in enumerate(pyreg.index)])
            eqn = '+'.join(['{}*{}'.format(x[:-1],x[:-1]) if x.endswith('2') else x 
                for x in eqn.split('+')])
            stata.equation = eqn.replace(' ','')
            stata.coefs = {'c{}'.format(i):float(p) 
                for i,p in enumerate(pyreg['Parameter'].values)}
            stata.std_err = {'c{}'.format(i):float(p) 
                for i,p in enumerate(pyreg['Std. Err.'].values)} 
            varlist = list(np.unique([x.strip() 
                for x in re.split('\*|\+',stata.equation) if not x.strip().startswith('c')]))
            if 'np.log(income)' in varlist:
                varlist.remove('np.log(income)')
                varlist+=['income']
            stata.varlist = varlist
            stata.coefs_sig = {c:3 for c,co in stata.coefs.items()}

        eq = stata.equation*1

        if individual:
            for term in eq.split('+'):
                stata_copy = copy.copy(stata)
                stata_copy.equation = term
                statas[block][short(term)] = stata_copy
        else:
            statas[block]['tot'] = stata

    return statas 

def calc_international_migrants(country_df,ac_shares,mignames_num):

    mignames_num_inint  = [v+'_inint' for v in mignames_num]
    mignames_num_outint = [v+'_outint' for v in mignames_num]

    # Merge international migration shares with migration rate dataset
    # And multiply the number by the fraction into each country
    df1                     = country_df.merge(ac_shares, how='inner', on='adm0_country_id')
    df1[mignames_num_inint] = df1[mignames_num].multiply(
                                 df1['perc_da_pb_closed'],axis='index')*nl.ac_beta

    # Save before summing
    if nl.save_everything:
        #pdb.set_trace()
        df1.drop(columns='geometry').to_csv('./{}_sourcereceptor.csv'.format(nl.run_name))

    # Group destinations and sum
    grpvars = ['year','ssp','model','dest']
    df1     = df1.groupby(grpvars)[mignames_num_inint].sum().reset_index()
    df1     = df1.rename(columns={"dest" : "adm0_country_id"})

    ## Merge migrant numbers back with main dataframe
    country_out = country_df.merge(df1, how='left',on=('adm0_country_id','year','ssp','model'))

    # Add out-migration number
    country_out[mignames_num_outint] = country_out[mignames_num]*nl.ac_beta

    return country_out


def proj(inp,countries,statas,ac_shares):
    """
    Wrap projection into function. 
    Move stata portion outside for more control over equation and bin.
    That way it's easy to run just one variable and/or just one bin
    """
    keys = list(statas.keys())
    uterms = {}
    proj_ctry = {}

    # Loop through selected model blocks
    for block in nl.block_num:
        print(block)
        proj_dfs = {}

        # Loop through pre-loaded stata table terms
        for term, stata in statas[block].items():
            inp_cp = inp.copy()

            #   Grid cell out-migration calc
            eval_block = stata.evaluate(sigLevel=0,
                T      = inp.Temperature, P      = inp.Precipitation,
                Tm1    = inp.Tm1,         Pm1    = inp.Pm1,
                T2m1   = inp.T2m1,        P2m1   = inp.P2m1,
                Tmean  = inp.Tmean,
                Ttrend = inp.Ttrend,      Ptrend = inp.Ptrend,
                Tanom  = inp.Tanom,       Panom  = inp.Panom,
                Tvar   = inp.Tvar,        Pvar   = inp.Pvar,
                income = np.exp(inp.gdppc),
                )

            inp_cp['mig_rate_nodif'] = eval_block
            inp_cp_base              = inp_cp.loc[inp_cp.year==nl.baseline_yr]
            inp_cp_base              = inp_cp_base.rename(columns={'mig_rate_nodif':'mig_rate_base'})
            mvar                     = ['lat','lon','adm1_state_id','ssp','model']
            inp_cp                   = inp_cp.merge(inp_cp_base[mvar+['mig_rate_base']],
                                                  on=mvar,how='left')
            inp_cp['mig_rate']       = inp_cp['mig_rate_nodif'] - inp_cp['mig_rate_base']
    
            proj_dfs[term]    = inp_cp


        # Combine terms into one dataframe so that there is only one aggregation step instead of many
        mignames = []
        for ti,(term,df) in enumerate(proj_dfs.items()):
            mig_cur = 'mig_rate_{}'.format(term)
            mignames.append(mig_cur)
            if ti==0: 
                proj_in = df.rename(columns={'mig_rate':mig_cur})
            else:
                proj_in[mig_cur] = df['mig_rate']
        mignames = list(np.unique(mignames))
        mignames_num = [n.replace('mig_rate','mig_num') for n in mignames]

        if nl.save_everything:
            #pdb.set_trace()
            proj_in.to_csv('./{}_full_05deg.csv'.format(nl.run_name))
        
        # Aggregate to country and merge with country shapefile
        wt_mean_vars = mignames + ['pop_total_rural','gdppc',
                                   'Temperature','Ttrend','Tvar','Tshock','Tanom','Tmean',
                                   'Precipitation','Ptrend','Pvar','Pshock','Panom']
        prod_vars    = [v+'_pop' for v in wt_mean_vars]

        # Note that pop_total_rural is the total population IN rural grid cells
        proj_in['adm0_country_id'] = proj_in['adm0_country_id'].astype(str)

        # Save gridded version here if desired

        if nl.const_gdp:
            if len(statas[0])==1:
                print('Saving gridded projection for fixgdp, mean model')
                proj_in.to_csv('./{}_fixgdp_grid_projection.csv'.format(nl.run_name))
        elif not nl.const_gdp:
            if len(statas[0])==1:
                print('Saving gridded projection for sspgdp, mean model')
                proj_in.to_csv('./{}_sspgdp_grid_projection.csv'.format(nl.run_name))
        print('Finished saving gridded')

        for v,pv in zip(wt_mean_vars,prod_vars):
            proj_in[pv] = proj_in[v]*proj_in['pop_total_rural']
        proj_ctry[block] = proj_in.groupby(
            ['year','adm0_country_id','ssp','model'])[prod_vars+['pop_total_rural']].sum()
            
         # Save sum of pop_total_rural (want sum not weighted mean)
        pop_total_rural = proj_ctry[block]['pop_total_rural']

        proj_ctry[block][wt_mean_vars] = proj_ctry[block][prod_vars].div(
            proj_ctry[block]['pop_total_rural'],axis=0)
        proj_ctry[block]['pop_total_rural'] = pop_total_rural # over write pop from above

        # Merge with shapefiles and convert to geodataframe
        proj_ctry[block] = proj_ctry[block].reset_index().merge(
            countries,on='adm0_country_id',how='left')

        proj_ctry[block] = gpd.GeoDataFrame(proj_ctry[block])

        # Calculate total migrant number and subsequently the international migration rate
        for mn, mnn in zip(mignames,mignames_num):
            proj_ctry[block][mnn] = proj_ctry[block][mn]*proj_ctry[block]['pop_total_rural']

        proj_ctry[block] = calc_international_migrants(proj_ctry[block],ac_shares,mignames_num)

    return proj_ctry

#   #   #   #   #   #   #   #   #   #
#   Load stata tables and project
#   #   #   #   #   #   #   #   #   #

print('Reading regression for full model')
statas_full = read_stata(blocks=nl.block_num, f=nl.stata_reg, individual=False)
print('Projecting full model...')
proj_ctry_full = proj(inp,countries,statas_full,ac_shares)
with open('{}/rep_full.pickle'.format(nl.out_dir),'wb') as handle:
    pickle.dump(proj_ctry_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Finished projecting full model.')

print('Reading regression for terms of model')
statas_term = read_stata(blocks=nl.block_num, f=nl.stata_reg, individual=True)
print('Projecting terms of model...')
proj_ctry_term = proj(inp,countries,statas_term,ac_shares)
with open('{}/rep_term.pickle'.format(nl.out_dir),'wb') as handle:
    pickle.dump(proj_ctry_term, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Finished projecting terms of model.')

