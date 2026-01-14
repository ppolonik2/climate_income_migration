"""
Adopts settings from run.bash
Sets run-specific settings, defined by the run name
Sets a number of otehr normally unchanged variables

# The step prompts the user to input several parameters including 
#   the Shared Socioeconomic Pathway (SSP) scenario,
#   migration dataset (jrc or desherb), 
#   fixed effect type (admin or cell), 
#   weighting type (unit or pop),
#   file date,
#   Abel & Cohen country shares option (beta1 or beta2). 
# These inputs determine the model specification
"""

import numpy as np
import os
import pdb

# Set name for equation; full run name set below based on inputs
eq_name        = '__EQ_NAME__'
Nreps          = __NREPS__
sample_frac    = __SAMPLE_FRAC__
sample_type    = '__SAMPLE_TYPE__'
const_gdp      = __CONST_GDP__
pass_test_only = __PASSOPT__

## Run using only climate model mean?
mean_opt      = bool(__MEAN_OPT__)

#  #  #  #  #  #  #  #  #  #  #  #  
#  These are normal default settings. Some of them are changed for specific runs 
#  #  #  #  #  #  #  #  #  #  #  #  

## Run using urban instead of rural data? (change for urban options below)
urban_opt       = False

# Whether to run just for the richest third of data points
richthird       = False

# Turn off regression weighting
no_weights      = False

# Shuffle T and P in time
shuffle_clim    = False

# Save all output - only on for main specification
save_everything = False

#  #  #  #  #  #  #  #  #  #  #  #  
#  Run-specific settings
#  #  #  #  #  #  #  #  #  #  #  #

# Set regression equation
if eq_name.endswith('quadT_linP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    save_everything=True

elif eq_name.endswith('quadT_linP_nogdp'):
    equation = 'T+T2+P+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_shuffleclim'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    shuffle_clim=True

elif eq_name.endswith('quadT_linP_lagmig'):
    equation = 'gdppc1+gdppc2+outmigm1+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_lingdp'):
    equation = 'T+T*gdppc1+T2+T2*gdppc1+P+P*gdppc1+gdppc1+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_noweights'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    no_weights = True

elif eq_name.endswith('quadT_linP_richthird'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    richthird = True

elif eq_name.endswith('cubT_linP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+T3+T3*gdppc1+T3*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    richthird = False

elif eq_name.endswith('linT_linP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_rep100'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_passonly'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_rep100_50perc'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_statrep100'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    
elif eq_name.endswith('quadT_linP_urban'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    urban_opt=True
    save_everything=True

elif eq_name.endswith('quadT_linP_withlag'):
    equation = 'gdppc1+gdppc2+T+Tm1+T*gdppc1+Tm1*gdppc1+T*gdppc2+Tm1*gdppc2+T2+T2m1+T2*gdppc1+T2m1*gdppc1+T2*gdppc2+T2m1*gdppc2+P+Pm1+P*gdppc1+Pm1*gdppc1+P*gdppc2+Pm1*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    
elif eq_name.endswith('quadT_linP_lagonly'):
    equation = 'gdppc1+gdppc2+Tm1+Tm1*gdppc1+Tm1*gdppc2+T2m1+T2m1*gdppc1+T2m1*gdppc2+Pm1+Pm1*gdppc1+Pm1*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_urban_admin'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_rural'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_rural_adminsamp'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_allmodels'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_TP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+T*P+T*P*gdppc1+T*P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_Ttrend'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+Ttrend+Ttrend*gdppc1+Ttrend*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_Ttrend_Tvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+Ttrend+Ttrend*gdppc1+Ttrend*gdppc2+Tvar+Tvar*gdppc1+Tvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_Ttrend_Tvar_Ptrend'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+Ttrend+Ttrend*gdppc1+Ttrend*gdppc2+Tvar+Tvar*gdppc1+Tvar*gdppc2+Ptrend+Ptrend*gdppc1+Ptrend*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_Ttrend_Tvar_Ptrend_Pvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+Ttrend+Ttrend*gdppc1+Ttrend*gdppc2+Tvar+Tvar*gdppc1+Tvar*gdppc2+Ptrend+Ptrend*gdppc1+Ptrend*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_Pvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_Pvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_Pvar_PPvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_quadP_Pvar_PPvar'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+P2+P2*gdppc1+P2*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+P*Pvar+P*Pvar*gdppc1+P*Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('TTmean_linP'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T*Tmean+T*Tmean*gdppc1+T*Tmean*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif (eq_name.endswith('quadT_linP_1990onward'))|(eq_name.endswith('quadT_linP_pre1990')):
    # Manually change the meaning of onward_1990 for pre/post1990
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_nosplit'):
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=True
    minimum_significance=0

elif eq_name.endswith('Ttrend_Tvar_Ptrend_Pvar'):
    # Important that this comes after quadT_quadP_...same thing
    equation = 'gdppc1+gdppc2+Ttrend+Ttrend*gdppc1+Ttrend*gdppc2+Tvar+Tvar*gdppc1+Tvar*gdppc2+Ptrend+Ptrend*gdppc1+Ptrend*gdppc2+Pvar+Pvar*gdppc1+Pvar*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0
    equation = 'gdppc1+gdppc2+T+T*gdppc1+T*gdppc2+T2+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=True
    minimum_significance=0

elif eq_name.endswith('quadT_linP_Tm1'):
    equation = 'gdppc1+gdppc2+T+Tm1+T*gdppc1+T*gdppc2+T2+T2m1+T2*gdppc1+T2*gdppc2+P+P*gdppc1+P*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('quadT_linP_Tm1_inc'):
    equation = 'gdppc1+gdppc2+T+Tm1+T*gdppc1+Tm1*gdppc1+T*gdppc2+Tm1*gdppc2+T2+T2m1+T2*gdppc1+T2m1*gdppc1+T2*gdppc2+T2m1*gdppc2+P+Pm1+P*gdppc1+Pm1*gdppc1+P*gdppc2+Pm1*gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0

elif eq_name.endswith('gdponly'):
    equation = 'gdppc1+gdppc2+constant'
    onward_1990=True
    no_split=False
    minimum_significance=0


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #
# Don't change below here
# Mostly legacy settings that are no longer changed
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #

# Set whether to use stata or python regression results (determines formatting)
reg_type      = 'python' # 'stata' or 'python'

# Set GDP extrapolation method (linear, constant, anything else does no extrapolation)
extrap_gdp = 'constant'

# Which model specs (columns) to use from stata output
# If using python output, gets overwritten below 
block_num     = [1]

# Whether to use mean GDPpc or not
mean_gdppc = False

# Set stata regression file name. Used as a template if reg_type is python
stata_reg_f   = 'stata_output_template.xlsx'

# Abel & Cohen beta option 
# estimated_beta_ipums with updated value from Jacopo (9/17/2024)
ac_beta = 0.119879508277765 

baseline_yr = 2015
end_yr      = 2050

# Income scaling (True=log income; False=raw income)
# This is set based on whether income coefficients correspond to log(income)
flag_log_inc =  True

if const_gdp: 
    prefix = 'fixgdp'
else:
    prefix = 'sspgdp'
run_name    = '{}_{}'.format(prefix,eq_name)

# Set the blocks for monte carlo
if reg_type=='python':
    block_num = np.arange(Nreps)

# Set paths
root_dir   = os.path.expanduser('~/climate_income_migration/')
dat_dir    = os.path.join(root_dir,'data')
out_dir    = os.path.join(dat_dir,'projections',run_name)
stata_reg  = os.path.join(dat_dir,'stata_output',stata_reg_f)
python_reg = os.path.join(out_dir,'reg')


