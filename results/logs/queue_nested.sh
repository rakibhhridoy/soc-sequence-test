#!/bin/zsh
# Full rerun with nested hyperparameter selection and blocked early stopping (2026-09-26).
# The mechanistic penalty is dropped: it was never active on the LUCAS panels, which carry
# no delta_max or next-step covariates, so its ablation was vacuous.
cd "$(dirname "$0")/../.."
L=results/logs/nested; mkdir -p $L
BLOCK=276477.3517754666      # mineral variogram range; the organic variogram has no sill
run() { echo "[$(date +%H:%M)] $1 start"; shift; "$@" > $L/$STEP.log 2>&1 && echo "[$(date +%H:%M)] ok" || echo "[$(date +%H:%M)] FAILED ($?)"; }
STEP=ablations_mineral; run $STEP python -u scripts/run_ablations.py --data data/processed/panel_mineral.npz --epochs 200 --folds 4 --ensemble 3
STEP=ablations_organic; run $STEP python -u scripts/run_ablations.py --data data/processed/panel_organic.npz --epochs 200 --folds 3 --ensemble 3 --block-size $BLOCK
STEP=bootstrap; run $STEP python -u scripts/bootstrap_comparison.py
STEP=dynamic; run $STEP python -u scripts/diagnose_dynamic.py
STEP=climate; run $STEP python -u scripts/climate_comparison.py
STEP=temporal; run $STEP python -u scripts/temporal_validation.py
STEP=robustness; run $STEP python -u scripts/robustness.py
STEP=breakdown; run $STEP python -u scripts/soil_breakdown.py
STEP=graph_mineral; run $STEP python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4
STEP=graph_organic; run $STEP python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 --block-size $BLOCK
STEP=tuning_summary; run $STEP python -u scripts/tune_model.py
STEP=figures; run $STEP python -u scripts/make_figures.py
rsync -a --exclude '._*' --exclude __pycache__ results scripts src manuscript ~/soc_backup/article/
echo "[$(date +%H:%M)] all done"
