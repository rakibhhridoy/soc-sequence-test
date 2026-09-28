#!/bin/zsh
# Resume of queue_nested.sh after the SSD disconnected during robustness (2026-09-27).
# Runs from the internal-disk copy; finished steps are not repeated and cached nested
# searches are reused. Results are synced back to the SSD at the end if it is mounted.
cd "$(dirname "$0")/../.."
L=results/logs/nested; mkdir -p $L
BLOCK=276477.3517754666      # mineral variogram range; the organic variogram has no sill
run() { echo "[$(date +%H:%M)] $1 start"; shift; "$@" > $L/$STEP.log 2>&1 && echo "[$(date +%H:%M)] ok" || echo "[$(date +%H:%M)] FAILED ($?)"; }
STEP=robustness; run $STEP python -u scripts/robustness.py
STEP=breakdown; run $STEP python -u scripts/soil_breakdown.py
STEP=graph_mineral; run $STEP python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4
STEP=graph_organic; run $STEP python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 --block-size $BLOCK
STEP=tuning_summary; run $STEP python -u scripts/tune_model.py
STEP=figures; run $STEP python -u scripts/make_figures.py
rsync -a --exclude '._*' --exclude __pycache__ results scripts src ~/soc_backup/article/
SSD="${SOC_SSD_COPY:-}"      # optional mirror of results/
if [ -d "$SSD" ]; then
  rsync -a --exclude '._*' --exclude __pycache__ results "$SSD/" && echo "[$(date +%H:%M)] synced results to SSD"
else
  echo "[$(date +%H:%M)] SSD not mounted; results remain in ~/soc_work/article"
fi
echo "[$(date +%H:%M)] all done"
