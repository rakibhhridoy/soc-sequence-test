#!/bin/zsh
# Review analyses of 2026-10-22, run in order after neighbour_features.py finishes.
cd ~/soc_work/article
L=results/logs/review_2026-10-22
log() { echo "[$(date '+%d %H:%M')] $*" >> $L/queue.log; }
run() { local name=$1; shift; log "$name start"; if "$@" > $L/$name.log 2>&1; then log "$name ok"; else log "$name FAILED"; fi; }
while pgrep -f "scripts/neighbour_features.py" > /dev/null; do sleep 60; done
log "neighbour_features finished"
run temporal_extend      python3 -u scripts/temporal_extend.py
run robustness_pool      python3 -u scripts/robustness.py --pool-only
run boosting_pool        python3 -u scripts/boosting_features.py --pool-only
run neighbour_features5  python3 -u scripts/neighbour_features.py
run block_sensitivity    python3 -u scripts/block_sensitivity.py --block-km 390 --replicates 3
log "all done"
