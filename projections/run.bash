#/bin/bash

declare -a runs=(#"gdptime_1990_quadT_linP_rural_adminsamp"
                 #"gdptime_1990_quadT_linP_rep100_50perc"
                 #"gdptime_1990_quadT_linP_Tm1_inc"
                 #"gdptime_1990_quadT_linP_lagmig"
                 #"gdptime_1990_quadT_linP_nogdp"
                 #"gdptime_1990_quadT_linP_lingdp"
                 #"gdptime_1990_linT_linP"
                 "gdptime_1990_quadT_linP"
                 #"gdptime_1990_quadT_linP_shuffleclim"
                 #"gdptime_1990_quadT_quadP"
                 #"gdptime_1990_quadT_linP_urban"
                 #"gdptime_1990_quadT_linP_noweights"
                 #"gdptime_1990_cubT_linP"
                 #"gdptime_1990_quadT_linP_richthird"
                 #"gdptime_1990_linT_linP"
                 #"gdptime_1990_quadT_linP_withlag"
                 #"gdptime_1990_quadT_linP_lagonly"

                 #"gdptime_1990_quadT_linP_passonly"
                 #"gdptime_1990_quadT_quadP"
                 #"gdptime_1990_quadT_quadP_TP"
                 #"gdptime_1990_quadT_quadP_Ttrend"
                 #"gdptime_1990_quadT_quadP_Ttrend_Tvar"
                 #"gdptime_1990_quadT_quadP_Ttrend_Tvar_Ptrend"
                 #"gdptime_1990_quadT_quadP_Ttrend_Tvar_Ptrend_Pvar"
                 #"gdptime_1990_quadT_linP_Pvar"
                 #"gdptime_1990_quadT_Pvar"
                 #"gdptime_1990_quadT_linP_Pvar_PPvar"
                 #"gdptime_1990_quadT_quadP_Pvar_PPvar"
                 #"gdptime_1990_TTmean_linP"
                 #"gdptime_1990_quadT_linP_1990onward"
                 #"gdptime_1990_quadT_linP_nosplit"
                 #"gdptime_1990_Ttrend_Tvar_Ptrend_Pvar"
)
#declare -a runs=("gdptime_extrap_quadT_quadP_TP"
#                 "gdptime_extrap_quadT_quadP_Ttrend_Tvar"
#                 "gdptime_extrap_quadT_quadP_Ttrend_Tvar_Ptrend"
#                 "gdptime_extrap_quadT_quadP_Ttrend_Tvar_Ptrend_Pvar"
#                 "gdptime_extrap_TTmean_linP"
#                 "gdptime_extrap_quadT_linP_1990onward"
#                 "gdptime_extrap_quadT_linP_nosplit"
#                 "gdptime_extrap_Ttrend_Tvar_Ptrend_Pvar"
#)

NREPS=1
SAMPLEFRAC=1
SAMPLETYPE="random"  # admin or random - don't recommend mixing samplefram with admin sampletype
MEANOPT="True"
PASSOPT="False"

declare -a gdpconst=("False" "True") 

STARTPATH=$(pwd)
OUTDIR="$STARTPATH/../data/projections"
for r in "${runs[@]}"
do
    for g in "${gdpconst[@]}"
    do
        if [ "$g" = "False" ]; then
            prefix=sspgdp_
        else
            prefix=fixgdp_
        fi

        echo "$r: gdp_const=$g"
        OUTPATH="/$OUTDIR/$prefix$r"
        mkdir $OUTPATH
        mkdir "$OUTPATH/reg"

        cp namelist.py namelist_run.py
        sed -i s/__EQ_NAME__/$r/g namelist_run.py
        sed -i s/__CONST_GDP__/$g/g namelist_run.py
        sed -i s/__NREPS__/$NREPS/g namelist_run.py
        sed -i s/__SAMPLE_FRAC__/$SAMPLEFRAC/g namelist_run.py
        sed -i s/__SAMPLE_TYPE__/$SAMPLETYPE/g namelist_run.py
        sed -i s/__MEAN_OPT__/$MEANOPT/g namelist_run.py
        sed -i s/__PASSOPT__/$PASSOPT/g namelist_run.py

        mv namelist_run.py $OUTPATH
        cp run.py $OUTPATH
        cp run_projections.py $OUTPATH
        cp import_data.py $OUTPATH
        cp regres.py $OUTPATH
    done
done

for r in "${runs[@]}"
do
    for g in "${gdpconst[@]}"
    do
        if [ "$g" = "False" ]; then
            prefix=sspgdp_
        else
            prefix=fixgdp_
        fi

        OUTPATH="/$OUTDIR/$prefix$r"
        echo "$r: gdp_const=$g"
        cd $OUTPATH
        python run.py
        cd $STARTPATH
    done
done



