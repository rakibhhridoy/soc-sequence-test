#!/bin/zsh
# Full rerun after the 2026-10-09 fix: the loader had read electrical conductivity (EC) as
# organic carbon for the 2015 and 2018 campaigns. Panels were rebuilt with OC before this
# queue; every result below is regenerated from them, in the order of README.md.
cd "$(dirname "$0")/../../.."
L=results/logs/rerun_2026-10-09
BLOCK=$(python -c "import sys; sys.path[:0]=['src','scripts']; import config, train; from run_ablations import load_npz; print(train.block_size_for(load_npz(config.DATA_PROCESSED/'panel_mineral.npz')))")
echo "[$(date '+%d %H:%M')] mineral block size $BLOCK"
run() { STEP=$1; shift; echo "[$(date '+%d %H:%M')] $STEP start"; "$@" > $L/$STEP.log 2>&1 && echo "[$(date '+%d %H:%M')] $STEP ok" || echo "[$(date '+%d %H:%M')] $STEP FAILED ($?)"; }
run selftest           python -u scripts/selftest.py
run ablations_mineral  python -u scripts/run_ablations.py --data data/processed/panel_mineral.npz --epochs 200 --folds 4 --ensemble 3
run ablations_organic  python -u scripts/run_ablations.py --data data/processed/panel_organic.npz --epochs 200 --folds 3 --ensemble 3 --block-size $BLOCK
run bootstrap          python -u scripts/bootstrap_comparison.py
run dynamic            python -u scripts/diagnose_dynamic.py
run climate            python -u scripts/climate_comparison.py
run temporal           python -u scripts/temporal_validation.py
run robustness         python -u scripts/robustness.py
run breakdown          python -u scripts/soil_breakdown.py
run boosting_features  python -u scripts/boosting_features.py
run boosting_deep      python -u scripts/boosting_features.py --deep
run campaign_proxy     python -u scripts/campaign_proxy.py
run climate_breakdown  python -u scripts/climate_breakdown.py
run noise_ceiling      python -u scripts/noise_ceiling.py
run point_change       python -u scripts/point_change.py
run tuning_summary     python -u scripts/tune_model.py
run figures            python -u scripts/make_figures.py
rsync -a --exclude '._*' --exclude __pycache__ --exclude data results scripts src manuscript ~/soc_backup/article/
# graph extension last: full-batch graphs exceed Apple GPU memory, so these run on the CPU
export SOC_DEVICE=cpu
run graph_mineral      python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4 --graphs geographic,covariate-space,none
run graph_organic      python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 --block-size $BLOCK
run graph_bootstrap    python -u scripts/graph_bootstrap.py
unset SOC_DEVICE
run figures_final      python -u scripts/make_figures.py
rsync -a --exclude '._*' --exclude __pycache__ --exclude data results scripts src manuscript ~/soc_backup/article/
echo "[$(date '+%d %H:%M')] all done"
