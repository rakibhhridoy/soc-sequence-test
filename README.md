# Measurement noise limits soil carbon forecasting and crediting across European monitoring points

Code and result tables for a test of a convolutional-recurrent hybrid against gradient
boosting for forecasting change in soil organic carbon (SOC), on LUCAS topsoil points
sampled in 2009, 2015 and 2018.

Authors: Md Rakib Hasan, Mst Anika Khatun Rupa, A. S. M. Mohiuddin
(Department of Soil, Water and Environment, University of Dhaka; Fermium Systems).

## What the study finds

Mineral soils, 16,496 observations from 8,275 points, spatially blocked validation with
hyperparameters selected inside each training fold, pooled over five replicates:

| | Skill against persistence |
|---|---|
| Convolutional-recurrent hybrid | 0.113 |
| Gradient boosting on summary statistics | 0.110 |
| Linear fit on the previous value (mean-reversion floor) | 0.047 |
| Ceiling set by measurement noise | 0.28 (0.24 to 0.37) |

- Hybrid and boosting are equivalent across space: the difference is 0.003 (95 % interval
  -0.011 to 0.019), inside a margin of 0.02.
- Removing the recurrent decoder or the learned encoder does not lower skill, and the
  Landsat series, about a third of the skill, are read as well by boosting.
- Successive changes at a point correlate at -0.54, so measurement noise is at least 94 % of
  the mean squared six-year change; pH and total nitrogen show the same signature, and the
  apparent dependence of change on the starting level vanishes when instrumented.
- At a later survey and at points withheld in space, pooled over five replicates, the hybrid
  beats mean reversion by 0.083 (0.041 to 0.108) and tuned boosting by 0.020 (-0.003 to 0.042).
- With ERA5-Land climate added, pooled over five replicates, the hybrid and tuned boosting
  differ by 0.007 (-0.017 to 0.029), and with 56 blocks of 390 km instead of 18 of 780 km the
  hybrid and boosting differ by 0.001 (-0.013 to 0.022).
- Giving boosting the mean previous carbon of each point's graph neighbours recovers about
  half of the geographic graph's gain.
- In carbon-crediting terms (EU carbon farming methodology under Regulation (EU) 2024/3012),
  the noise in one remeasured sample is about 15 times the six-year gain of the 4 per 1000
  target, detecting that gain needs about 2,100 resampled locations, a 70 % one-sided
  confidence rule credits three in ten projects with no true change, and enrolment on a low
  baseline credits a reversion gain six times the target (`scripts/carbon_credit_noise.py`).

**Versions 2.0.0 to 2.2.2 are invalid.** Their loader read electrical conductivity as
organic carbon for the 2015 and 2018 campaigns. Version 3.0.0 fixes the loader and reruns
every result. Version 3.1.0 adds the climate comparison and the forward-in-time
design at new places over five replicates, a check on the block size, and neighbour features
for the tabular models. Version 3.2.0 adds the carbon-crediting analysis
and redraws the figures to print widths.

`results/tables/` holds every number reported in the article.

## Data

The LUCAS topsoil data are licensed to the recipient and are not redistributed here. Request
the 2009, 2015 and 2018 topsoil campaigns from the European Soil Data Centre (ESDAC); see
`data/LUCAS_HOWTO.md`. Landsat covariates are exported through Google Earth Engine and
ERA5-Land climate from the Copernicus Climate Data Store. The scripts rebuild the analysis
panel from those files, so anyone with ESDAC access can reproduce every table.

## Reproducing the results

```
conda env create -f environment.yml
python scripts/lucas_repeat_count.py        # match campaigns, count repeated points
python scripts/export_covariates.py         # Landsat monthly series (Earth Engine)
python scripts/extract_era5land.py          # ERA5-Land monthly climate
python scripts/build_panels.py              # mineral and organic panels
python scripts/build_climate_panel.py       # panels with climate channels
python scripts/run_ablations.py --data data/processed/panel_mineral.npz --epochs 200 --folds 4 --ensemble 3
python scripts/run_ablations.py --data data/processed/panel_organic.npz --epochs 200 --folds 3 --ensemble 3 --block-size 779810.8924983748
python scripts/bootstrap_comparison.py      # paired block bootstrap, spatial and temporal
python scripts/diagnose_dynamic.py          # covariate series replaced by zeros or noise
python scripts/climate_comparison.py        # adds ERA5-Land climate
python scripts/temporal_validation.py       # forward in time, same points
python scripts/robustness.py                # five replicates, mean-reversion floor, new-places design
python scripts/soil_breakdown.py            # land use, SOC tertiles, change map
python scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4 --graphs geographic,covariate-space,none
python scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3 --block-size 779810.8924983748
python scripts/graph_bootstrap.py           # paired intervals for the graph extension (needs the none and geographic runs)
python scripts/campaign_proxy.py            # do the Landsat series act as a proxy for the campaign?
python scripts/boosting_features.py         # stronger boosting: timing features and nested selection (after robustness.py)
python scripts/climate_breakdown.py         # skill by Köppen climate zone (needs the climate panel)
python scripts/noise_ceiling.py             # ceiling on skill set by measurement noise
python scripts/point_change.py              # noise share of change, instrumented level dependence, strata, controls
python scripts/tune_model.py                # summary of the nested hyperparameter selection
python scripts/climate_tuned.py             # tuned boosting with climate, single assignment
python scripts/climate_replicates.py        # climate comparison over the five replicates
python scripts/temporal_extend.py           # new-places design for replicates 3 and 4
python scripts/robustness.py --pool-only    # rebuild pooled tables from saved predictions
python scripts/boosting_features.py --pool-only
python scripts/neighbour_features.py        # graph neighbours' previous carbon as tabular features
python scripts/block_sensitivity.py --block-km 390 --replicates 3
python scripts/carbon_credit_noise.py      # measurement noise in carbon-crediting terms
python scripts/make_figures.py
```

`python scripts/selftest.py` checks the machinery on synthetic data. Hyperparameters are
selected inside every training fold (`src/tuning.py`) and cached in
`results/tables/nested_tuning.json`. On an 8 GB machine the full-batch mineral graph runs
exceed Apple GPU memory; set `SOC_DEVICE=cpu`.

## Layout

```
src/       importable modules: panel building, blocking, models, training, tuning, evaluation
scripts/   entry points; each writes to results/
results/   tables, figures and run logs
data/      where the inputs come from (no data are tracked)
```

## Licence

MIT for the code. The LUCAS data remain under their own licence.
