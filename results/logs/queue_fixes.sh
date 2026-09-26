#!/bin/zsh
# Full rerun after the texture and mineral-classing fixes (2026-09-26).
cd "/Volumes/SSD Rx/Research/Others/Seminar/article"
L=results/logs/fixes; mkdir -p $L
run() { echo "[$(date +%H:%M)] $1 start"; shift; "$@" > $L/$STEP.log 2>&1 && echo "[$(date +%H:%M)] ok" || echo "[$(date +%H:%M)] FAILED ($?)"; }
STEP=climate_panel; run $STEP python -u scripts/build_climate_panel.py
STEP=ablations_mineral; run $STEP python -u scripts/run_ablations.py --data data/processed/panel_mineral.npz --epochs 200 --folds 4 --hidden 32 --rnn-hidden 32 --lr 0.003 --dropout 0.1 --ensemble 3
STEP=ablations_organic; run $STEP python -u scripts/run_ablations.py --data data/processed/panel_organic.npz --epochs 200 --folds 3 --hidden 32 --rnn-hidden 32 --lr 0.003 --dropout 0.1 --ensemble 3
STEP=bootstrap; run $STEP python -u scripts/bootstrap_comparison.py
STEP=dynamic; run $STEP python -u scripts/diagnose_dynamic.py
STEP=climate; run $STEP python -u scripts/climate_comparison.py
STEP=temporal; run $STEP python -u scripts/temporal_validation.py
STEP=robustness; run $STEP python -u scripts/robustness.py
STEP=breakdown; run $STEP python -u scripts/soil_breakdown.py
STEP=figures; run $STEP python -u scripts/make_figures.py
STEP=graph_mineral; run $STEP python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4
STEP=graph_organic; run $STEP python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3
rsync -a --exclude '._*' --exclude __pycache__ results scripts src ~/soc_backup/article/
echo "[$(date +%H:%M)] all done"
