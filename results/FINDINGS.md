# Findings (2026-09-24) -- SUPERSEDED

> **Superseded on 2026-09-25.** These results were produced before a data bug was found
> (cation exchange capacity missing for half the panel, which gave NaN inputs and partly
> untrained networks). They are kept as a record only. The corrected results are in
> `STATUS.md`, and under them the falsification condition is **not** met.

The falsification condition fixed in advance is met. Gradient boosting on hand-engineered
temporal summary statistics outperforms the hybrid architecture on LUCAS mineral soils
under spatially blocked validation, and no ablation or tuning reverses that.

`tables/ablations.csv`, 16,200 mineral-soil observations from 8,100 points, 4 spatially
blocked folds, 200 epochs, 3-model ensembles, target log SOC.

| Model | skill vs persistence | RMSE (log SOC) | CCC (change) | 90 % coverage |
|---|---|---|---|---|
| gradient boosting (summary statistics) | **+0.265** | 0.510 | 0.663 | -- |
| 1. no recurrent decoder | +0.173 | 0.573 | 0.621 | 0.886 |
| 2. no learned encoder | +0.164 | 0.580 | 0.617 | 0.888 |
| 4. no static embedding | +0.155 | 0.586 | 0.607 | 0.885 |
| full model (LSTM, tuned) | +0.154 | 0.586 | 0.606 | 0.890 |
| 5. lambda = 0 | +0.154 | 0.586 | 0.606 | 0.890 |
| 6. no previous observation | +0.152 | 0.588 | 0.604 | 0.891 |
| 3. GRU in place of LSTM | +0.151 | 0.588 | 0.604 | 0.889 |
| persistence | 0.000 | 0.693 | 0.000 | -- |

## What the numbers say

**The architecture loses to the tabular baseline by a wide margin**, 0.154 against 0.265.
The preprint stated the test in advance: the framework is refuted if gradient boosting on
summary statistics matches the full architecture under blocked validation. It does not
merely match it.

**Every added component subtracts.** Removing the recurrent decoder improves skill to
0.173, removing the learned encoder to 0.164. The ordering is the opposite of the design
argument. All eight deep variants fall within 0.151 to 0.173, a band narrow enough that
the components are not distinguishable from one another.

**Tuning does not rescue it.** Twelve configurations on a separate blocked split scored
0.185 to 0.213, and the winner was the smallest network tested. Carrying it into the full
evaluation with 3-model ensembles moved skill from 0.152 to 0.154. When extra capacity
does not help, the constraint is the data rather than the model size.

**The models are not broken.** Every one beats persistence, prediction intervals are close
to calibrated at 0.885 to 0.891 against a nominal 0.90, and optimism from random splitting
is only 3 %, so the blocking is working and the comparison is sound.

## Reading it against the noise ceiling

The panel diagnostics put the attainable skill at roughly 0.48, set by a within-point
noise floor of 5.78 g/kg. Gradient boosting reaches about 55 % of that; the hybrid reaches
about 32 %. The task is therefore neither impossible nor solved, and the gap between the
two methods is real rather than an artifact of both saturating.

## Why this is a result rather than a failure

At LUCAS sampling density, with three surveys nine years apart and Landsat covariates, the
sequence structure a recurrent decoder exists to exploit is not recoverable. Two
measurements per point separated by years do not constrain a model of monthly dynamics,
and the measured change is itself partly noise: consecutive changes correlate at -0.268,
the signature of mean reversion.

The binding constraint is the monitoring network, not the architecture. Adding capacity to
the model cannot substitute for observations that were never taken, and a soil-carbon
monitoring programme intending to support machine learning should spend on resampling
frequency before it spends on model complexity.

## Limits of this result

1. One region and one survey programme. It does not show that sequence models fail for
   soil carbon everywhere, only that they fail here.
2. Landsat only, at 30 m. Denser or finer covariates might carry more.
3. Three time points. An architecture built for long sequences is being asked to work with
   two intervals per point.
4. Mineral soils only. Organic soils, 3 % of the panel, are excluded from this table and
   need their own analysis.
5. The baseline used library defaults while the network was tuned, which favours the
   network. The conclusion is therefore conservative.
