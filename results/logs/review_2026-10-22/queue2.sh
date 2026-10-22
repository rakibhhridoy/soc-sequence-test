#!/bin/zsh
# Climate over five replicates, run after queue.sh reports "all done".
cd ~/soc_work/article
L=results/logs/review_2026-10-22
log() { echo "[$(date '+%d %H:%M')] $*" >> $L/queue.log; }
until grep -q "all done" $L/queue.log 2>/dev/null; do sleep 120; done
log "climate_replicates start"
if python3 -u scripts/climate_replicates.py > $L/climate_replicates.log 2>&1; then log "climate_replicates ok"; else log "climate_replicates FAILED"; fi
log "queue2 done"
