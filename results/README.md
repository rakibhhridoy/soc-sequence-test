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
| `organic_boosting_with_previous.csv` | computed on the organic folds | boosting with the previous value, organic soils |
| `nested_tuning.json`, `nested_tuning_summary.csv` | `src/tuning.py`, `tune_model.py` | hyperparameters selected inside each training fold |
| `lucas_repeat_counts.csv` | `lucas_repeat_count.py` | points measured in each campaign |
| `ablations_synthetic.csv` | `run_ablations.py --synthetic` | machinery check on synthetic data only |

Superseded tables are kept for the record and are not cited in the article:
`*_lr001_do02*` (a configuration used before hyperparameter selection),
`ablations_organic_untracked_settings.csv`, `graph_*_LEAKY.csv` (a graph run in which each
point's later observation could reach its earlier one), and `tuning.json` /
`tuning_prefix.json` (a single search superseded by the nested selection).

`predictions/` is not tracked, because it is keyed to LUCAS point identifiers.
