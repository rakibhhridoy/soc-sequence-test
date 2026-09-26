#!/bin/zsh
set -e
cd "/Volumes/SSD Rx/Research/Others/Seminar/article"
L=results/logs
echo "[$(date)] organic ablations start"
python -u scripts/run_ablations.py --data data/processed/panel_organic.npz \
  --epochs 200 --folds 3 --hidden 32 --rnn-hidden 32 --lr 0.003 --dropout 0.1 --ensemble 3 > $L/ablations_organic_tuned.log 2>&1
echo "[$(date)] temporal start"
python -u scripts/temporal_validation.py > $L/temporal_tuned.log 2>&1
rsync -a --exclude '._*' --exclude '__pycache__' results scripts ~/soc_backup/article/
echo "[$(date)] all done"
