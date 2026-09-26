#!/bin/zsh
# Graph extension (STATUS.md item 2). Organic first: 268 points, cheap. Mineral is full-batch over 16,200 nodes.
set -e
cd "/Volumes/SSD Rx/Research/Others/Seminar/article"
L=results/logs
echo "[$(date)] graph organic start"
python -u scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 > $L/graph_organic_causal.log 2>&1
cp -p results/tables/graph_organic.csv ~/soc_backup/article/results/tables/ 2>/dev/null || true
echo "[$(date)] graph mineral start"
python -u scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4 > $L/graph_mineral_causal.log 2>&1
cp -p results/tables/graph_mineral.csv ~/soc_backup/article/results/tables/ 2>/dev/null || true
cp -p src/graphs.py ~/soc_backup/article/src/; cp -p scripts/run_graph.py ~/soc_backup/article/scripts/
echo "[$(date)] all done"
