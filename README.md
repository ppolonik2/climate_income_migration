# climate_income_migration

This is the repository for code related to the project:
Income strongly mediates climate-driven migration

# Documentation
* environment.yml contains the conda environment information about this project
    * This is not necessarily the minimum working version, but is what is used to run these scripts


* ./projections/ contains regression and projection code
    * To run the code, use 'bash run.bash'
    * run.bash contains a few options and is the file to run to execulte all regressions and projections
    * In run.bash, set the names of the desired runs and some options:
        * NREPS sets the number of repeated runs (preferred: 1)
        * SAMPLEFRAC sets the fraction to sample in each repeated run (preferred: 1)
        * SAMPLETYPE determines how to do the sampling (does nothing when SAMPLEFRAC=1)
        * MEANOPT sets whether to run projections using the ensemble mean climate
        * PASSOPT robustness option to run projections using (keep False)
        * These settings are adopted in namelist.py
    * namelist.py contains settings, including the equations that are defined based on the name set in run.bash
        * So there must be an entry in namelist.py corresponding to each run name in run.bash
        * Also contains a few other settings that should generally not be changed
    * When running, the files in ./projections/ are copied into ./data/projections/ and everything is run from there
        * This ensures that there is a record of exactly what was run (e.g. namelist)
        * Output is also saved there - regressions in ./projections/<run_name>/reg/
        * A separate directory is created for the fixed and time-varying income cases
        * That means regressions are run twice, but it doesn't add much time


* ./analysis contians plotting scripts
    * the first data figure with contours and parabolas is made using contours_fig_wparabolas.py
    * the map figure is made by proj_figs_wgridonly.py
    * the uncertainty figure is made using uncertainty_summary.py
    * the SI histograms curves for T and P are made using hist_parabs_TP.py
    * the SI cumulative distribution and map of grid-cell regressions are created in robustness_test.py
    * the SI histograms of different random sampling methods is created in sample_method_comparison.py
    * the SI figure of different specification projects (number markers) is created in spec_uncertainty.py
    * the SI 2D histogram of residuals is created in check_residuals.py
    * the SI tables are made in build_interactions.py

