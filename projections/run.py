import sys
import subprocess
import shutil
import os
import shutil
import namelist_run as nl

def yes_no_check(yn):
    if len(yn)>0:
        yn0 = yn.lower()[0]
    else:
        yn0 = ''
    ynok = yn0 in ['y','n']
    return ynok, yn0

# If using python regressions, run the regressions
if nl.reg_type=='python':
    print('Running python regressions')
    subprocess.run(['python','regres.py'])
    print('Finished python regressions')

# Then run the projections
print('Running projections')
subprocess.run(['python','run_projections.py'])
print('Finished projections')


