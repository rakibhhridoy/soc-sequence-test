# Results

Every number in the article comes from a table here. Tables are written by the scripts
named below and are rebuilt by rerunning them.

| Table | Script | Content |
|---|---|---|
| `ablations_mineral.csv`, `ablations_organic.csv` | `run_ablations.py` | full model, five ablations, boosting, persistence, random k-fold reference |
| `bootstrap_comparison.csv`, `bootstrap_per_fold.csv` | `bootstrap_comparison.py` | paired block-bootstrap intervals, spatial and same-point temporal |
| `robustness.csv`, `robustness_per_replicate.csv` | `robustness.py` | five replicates, mean-reversion floor, equivalence test, new-places temporal design |
| `dynamic_diagnostic.csv` | `diagnose_dynamic.py` | covariate series replaced by zeros or noise |
| `climate_comparison.csv` | `climate_comparison.py` | adding ERA5-Land climate |
| `temporal_validation.csv` | `temporal_validation.py` | forward in time at the same points |
| `breakdown_*.csv`, `change_by_land_use.csv` | `soil_breakdown.py` | simple baselines, land use, SOC tertiles, observed change |
| `graph_mineral.csv`, `graph_organic.csv` | `run_graph.py` | graph extension and its edge-free control |
| `graph_mineral_bootstrap.csv` | `graph_bootstrap.py` | paired block-bootstrap intervals for the graph extension, mineral soils |
| `campaign_proxy.csv` | `campaign_proxy.py` | whether the Landsat series carry the campaign: boosting with the campaign as an input, and the new-places design without the series |
| `boosting_features.csv` | `boosting_features.py` | boosting with timing features and nested selection, against the hybrid across space and at a later survey |
| `boosting_features_tuning.json` | `boosting_features.py` | the configuration selected in every training fold |
| `boosting_features_deep.csv`, `boosting_features_deep_tuning.json` | `boosting_features.py --deep` | the same with the tuning grid extended to depths 10 and 12 |
| `breakdown_climate.csv` | `climate_breakdown.py` | skill of the hybrid, boosting and the mean-reversion floor by Köppen climate zone |
| `noise_ceiling.csv` | `noise_ceiling.py` | error variance of one measurement from successive changes, the ceiling it sets on skill, and each model's share of the attainable reduction in error |
| `point_change.csv` | `point_change.py` | share of measured change that is noise, slope of change on the starting level (naive, mean of two, instrumented by 2009), ceiling by stratum, pH and nitrogen controls |
| `nested_tuning.json`, `nested_tuning_summary.csv` | `src/tuning.py`, `tune_model.py` | hyperparameters selected inside each training fold |
| `lucas_repeat_counts.csv` | `lucas_repeat_count.py` | points measured in each campaign |

Every table was regenerated in version 3.0.0 after the loader was found to read
electrical conductivity as organic carbon for 2015 and 2018. The invalid outputs, and the
earlier superseded tables, are kept locally and are not tracked.

`predictions/` is not tracked, because it is keyed to LUCAS point identifiers.
