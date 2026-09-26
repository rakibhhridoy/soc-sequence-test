#!/bin/zsh
# Queue from STATUS.md "Unfinished" items 1 and 2, run 2026-09-26.
set -e
cd "/Volumes/SSD Rx/Research/Others/Seminar/article"
L=results/logs
echo "[$(date)] main table start" 
python -u scripts/run_ablations.py --data data/processed/panel_mineral.npz \
  --epochs 200 --folds 4 --hidden 32 --rnn-hidden 32 --lr 0.003 --dropout 0.1 --ensemble 3 \
  > $L/ablations_mineral_2026-09-26.log 2>&1
echo "[$(date)] graph mineral start"
python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4 \
  > $L/graph_mineral_2026-09-26.log 2>&1
echo "[$(date)] graph organic start"
python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 \
  > $L/graph_organic_2026-09-26.log 2>&1
echo "[$(date)] backup"
rsync -a --exclude '._*' --exclude '__pycache__' ./ ~/soc_backup/article/
echo "[$(date)] all done"
