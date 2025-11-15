import numpy as np
import pandas as pd
from linearmodels import PanelOLS
pd.options.display.float_format = '{:10,.5f}'.format
import pdb


exps = [
    'gdptime_1990_quadT_linP_nogdp',
    'gdptime_1990_linT_linP',
    'gdptime_1990_quadT_linP_lingdp',
    'gdptime_1990_quadT_linP',
    'gdptime_1990_quadT_quadP',
]

regs_admin = []
regs_ctry  = []

admin_cols = ['Parameter','Std. Err.',     'P-value',    ]
ctry_cols  = ['Parameter','Std. Err. Ctry','P-value Ctry']
other_cols = ['Nobs','R2 inclusive','R2 within','mean dev var','sd dev var']
admin_cols += other_cols
ctry_cols  += admin_cols

tables_admin = {}
tables_ctry  = {}
for ei,exp in enumerate(exps):
    reg = pd.read_csv('../data/projections/sspgdp_{}/reg/rep000.csv'.format(exp),index_col=0)
    table_admin = reg[admin_cols]
    table_ctry  = reg[ctry_cols]

    tables_admin[ei] = table_admin
    tables_ctry[ei]  = table_ctry

panel_merged_admin = pd.concat(tables_admin,axis=1)
panel_merged_ctry  = pd.concat(tables_ctry,axis=1)

def i_sort(df):
    index_order = [
        'T','T*gdppc1','T*gdppc2',
        'T2','T2*gdppc1','T2*gdppc2',
        'P','P*gdppc1','P*gdppc2',
        'P2','P2*gdppc1','P2*gdppc2',
        'gdppc1','gdppc2','outmigm1','constant'
    ]
    index_avail = [i for i in index_order if i in df.index]
    return(df.loc[index_avail])

panel_merged_admin = i_sort(panel_merged_admin)
panel_merged_ctry  = i_sort(panel_merged_ctry)

panel_merged_admin.to_csv('../data/regres/build_reg_admin_id.csv')
panel_merged_ctry.to_csv('../data/regres/build_reg_adm0_country_id.csv')

# Repeat for urban
exp = 'gdptime_1990_quadT_linP_urban'
reg = pd.read_csv('../data/projections/sspgdp_{}/reg/rep000.csv'.format(exp),index_col=0)
tables_urban = {}
tables_urban['urban_admin']   = reg[admin_cols]
tables_urban['urban_country'] = reg[ctry_cols]
panel_merged_urban = pd.concat(tables_urban,axis=1)
panel_merged_urban = i_sort(panel_merged_urban)
panel_merged_urban.to_csv('../data/regres/build_reg_urban.csv')

# One more time for lagged migration (which didn't run projections, but the regressions worked)
exp = 'gdptime_1990_quadT_linP_lagmig'
reg = pd.read_csv('../data/projections/sspgdp_{}/reg/rep000.csv'.format(exp),index_col=0)
tables_lagmig = {}
tables_lagmig['lagmig_admin']   = reg[admin_cols]
tables_lagmig['lagmig_country'] = reg[ctry_cols]
panel_merged_lagmig = pd.concat(tables_lagmig,axis=1)
panel_merged_lagmig = i_sort(panel_merged_lagmig)

lagmig_simple = {}
lagmig_simple['with lag']    = panel_merged_lagmig['lagmig_admin'].iloc[:,[0,2]]
lagmig_simple['without lag'] = panel_merged_admin[4].iloc[:,[0,2]]
lagmig_simple = i_sort(pd.concat(lagmig_simple,axis=1))
#panel_merged_lagmig.to_csv('../data/regres/build_reg_lagmig.csv')


