# Deep sequence learning and gradient boosting for forecasting soil organic carbon change in Europe

Code and result tables for a test of a convolutional-recurrent hybrid against gradient
boosting for forecasting change in soil organic carbon (SOC), on LUCAS topsoil points
sampled in 2009, 2015 and 2018.

Authors: Md Rakib Hasan, Mst Anika Khatun Rupa, A. S. M. Mohiuddin
(Department of Soil, Water and Environment, University of Dhaka; Fermium Systems).

## What the study finds

Mineral soils, 16,462 observations from 8,339 points, spatially blocked validation with
hyperparameters selected inside each training fold, pooled over five replicates:

| | Skill against persistence |
|---|---|
| Gradient boosting on summary statistics | 0.254 |
| Convolutional-recurrent hybrid | 0.241 |
| Linear fit on the previous value (mean-reversion floor) | 0.169 |

- The hybrid does not outperform boosting: the 95 % interval on the difference is
  -0.031 to 0.003.
- Removing its recurrent decoder or learned encoder raises skill slightly.
- About two thirds of either method's skill is regression towards the mean.
- Forward in time, at points withheld in space, the hybrid leads boosting by 0.049
  (0.017 to 0.082), but its margin over the mean-reversion floor (0.030) has an interval
  reaching below zero.

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
python scripts/run_ablations.py --data data/processed/panel_organic.npz --epochs 200 --folds 3 --ensemble 3 --block-size 276477.3517754666
python scripts/bootstrap_comparison.py      # paired block bootstrap, spatial and temporal
python scripts/diagnose_dynamic.py          # covariate series replaced by zeros or noise
python scripts/climate_comparison.py        # adds ERA5-Land climate
python scripts/temporal_validation.py       # forward in time, same points
python scripts/robustness.py                # five replicates, mean-reversion floor, new-places design
python scripts/soil_breakdown.py            # land use, SOC tertiles, change map
python scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4 --graphs geographic,covariate-space,none
python scripts/graph_bootstrap.py           # paired intervals for the graph extension (needs the none and geographic runs)
python scripts/campaign_proxy.py            # do the Landsat series act as a proxy for the campaign?
python scripts/boosting_features.py         # stronger boosting: timing features and nested selection (after robustness.py)
python scripts/climate_breakdown.py         # skill by Köppen climate zone (needs the climate panel)
python scripts/noise_ceiling.py             # ceiling on skill set by measurement noise
python scripts/tune_model.py                # summary of the nested hyperparameter selection
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
