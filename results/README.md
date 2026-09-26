# Results

## `tables/ablations_synthetic.csv`

Synthetic data only. This verifies the machinery; it says nothing about soil carbon.

The panel is generated so that change responds to a recent-window mean of one dynamic
channel plus one static covariate, with strong persistence and spatially clustered
points. Run: `python scripts/run_ablations.py --synthetic --epochs 200 --folds 3`.

What it establishes:

- The pipeline completes end to end on the Apple GPU: blocked folds, early stopping,
  ensembles, six ablations, baselines and the optimism reference.
- Withholding the previous observation (ablation 6) raises RMSE from 0.59 to 2.18, which
  is why that value is an input. It is known when a forecast is made.
- Prediction intervals are close to calibrated, with 90 % nominal coverage landing at
  0.87 to 0.90 across variants.
- Random splitting looks 3 % better than blocked splitting, so the optimism reference
  works, though the gap is small because the generator's spatial clustering is mild.

What it must not be read as:

- The simpler variants score best here, with ablations 1 and 2 the only ones beating
  persistence at all. That follows from how the data were generated: the driver is a
  windowed mean, which hand-engineered summary statistics capture almost exactly, so
  there is nothing for a learned encoder or a recurrent decoder to add. A generator
  cannot tell anyone whether real soil carbon has structure worth learning.
- Absolute skill is meaningless here. Persistence is near-unbeatable by construction
  because the synthetic change signal is small against the standing stock.

The same table on the real LUCAS panel is the actual experiment, and the falsification
condition fixed in advance applies to that run alone.
