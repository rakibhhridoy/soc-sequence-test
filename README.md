# From framework to article

The preprint in `../preprint/` is frozen. It states an architecture and an
evaluation protocol fixed in advance, and reports no results. This folder turns it into
an empirical article by running that protocol.

Because the protocol was published first, its terms are binding. A result that
contradicts it is reportable, and changing the protocol to fit the data is not.

## Decisions taken (2026-09-23)

| | |
|---|---|
| Development data | EU LUCAS topsoil survey, resampled points across 2009 / 2015 / 2018 |
| Relation to preprint | Preprint supplies Introduction and Methods; this adds Data, Results, Discussion |
| Bangladesh | Discussion only. No Bangladeshi data enters the study, and no transfer claim is made (decision 2026-09-24) |
| Framework | PyTorch, on the Mac's Apple GPU (MPS backend) |
| Manuscript | `manuscript/`, started as a copy of the frozen preprint source |
| Dynamic sensor | Landsat 5/7/8/9 throughout, so every round is treated identically |
| Target | log SOC concentration; stock reported separately where bulk density exists |
| Covariate window | 60 months ending December of the year before each survey |
| Observations | 2015 and 2018 rounds as targets, with the preceding round as `y_prev` |

## Why LUCAS rather than Bangladesh data alone

The architecture needs a target measured repeatedly at identifiable locations. The
Bangladesh holdings give 9 sites at 2 time points, which cannot train a sequence model
and would only confirm the preprint's own failure condition. LUCAS resamples thousands
of points on a three-year cycle, so it can carry training, spatial blocking and forward
temporal validation. Bangladesh then becomes the transfer test, which is the harder and
more interesting question, and the one the preprint's limitations section flags.

Sentinel-2 begins only in 2015, so using it for the later rounds and Landsat for the
earlier ones would change sensor partway through the series, and any apparent temporal
signal could then be a sensor artefact. Landsat carries all rounds on one record
instead, at coarser resolution.

The risk to state plainly in the paper: LUCAS is European cropland and grassland, so a
model fitted there has no guaranteed validity on Bangladeshi floodplain paddy. That is
the hypothesis under test, not a defect to hide.

## Data integrity rules

1. `../../../SOC/data/SOC_Properties_40Years_1985_2025.csv` must never be a training
   target. Its yearly soil properties are interpolated between the 1985 and 2025
   surveys, so consecutive years repeat identical values, and its satellite-derived SOC
   column is the output of a ridge regression on the same satellite indices used as
   predictors. Training on it would learn the interpolation and score well for the wrong
   reason.
2. Raw downloads land in `data/raw/` and are never edited in place. Everything
   downstream is rebuilt by a script from `data/raw/`.
3. Every processed file records the script and date that produced it.

## Layout

```
article/
  manuscript/   LaTeX, started from the frozen preprint
  data/raw/     untouched downloads (LUCAS, exported covariates)
  data/interim/ intermediate build products
  data/processed/ model-ready tables
  src/          importable modules (see src/README.md)
  scripts/      entry points that produce something in results/
  notebooks/    exploration only; nothing depends on a notebook
  results/      figures, tables, trained models
```

## Status

- [x] Protocol code written and self-tested (`scripts/selftest.py`, 24 checks).
      Blocking, metrics, architecture, losses and baselines all run; the GRU holds
      exactly 0.750 of the LSTM's recurrent parameters and the upstream parameter count
      is identical either way, so the controlled comparison holds in code.
- [x] LUCAS 2009, 2015 and 2018 loaded; 8,368 cropland and grassland points measured in
      all three rounds (`results/tables/lucas_repeat_counts.csv`).
- [x] Panel diagnostics (`results/PANEL_DIAGNOSTICS.md`): log target, mineral and organic
      soils separated, and an attainable skill ceiling of about 0.48 set by the noise floor.
- [ ] LUCAS 2022 is listed on ESDAC but has neither a download nor a request form, so the
      soil module appears unreleased. Ask ec-esdac@ec.europa.eu whether it is coming.
- [x] Covariate extraction (`src/covariates.py`): Landsat 5/7/8/9, cloud and shadow
      masked, monthly medians, with NDVI, NDWI and MNDWI as channels, so the water regime
      enters the model rather than being masked away.
- [x] LUCAS ingestion (`src/data_lucas.py`) and the repeat-point count
      (`scripts/lucas_repeat_count.py`), tested against a synthetic fixture with the
      differing column spellings the real releases use. Runs as soon as files land.
- [ ] LUCAS-scale extraction goes through `covariates.export_monthly_series`, an Earth
      Engine batch export. Point-by-point sampling costs ~3 s per month, which is fine for
      nine sites and impossible for thousands. That function is written but UNTESTED.
- [x] Training loop (`src/train.py`): blocked cross-validation, early stopping on a
      blocked validation fold, deep ensembles, and standardisation fitted on the training
      fold alone so no test information leaks in.
- [x] The five ablations fixed in the protocol (`scripts/run_ablations.py`), with persistence and
      gradient-boosting baselines and a labelled random-k-fold optimism reference.
- [x] Verified end to end on synthetic data (`results/README.md`): six ablations,
      baselines, optimism reference, calibrated intervals. Machinery only.
- [x] Run on the real LUCAS panel. The first run (`results/FINDINGS.md`, now superseded)
      was invalidated by a data bug; see `results/STATUS.md` for the corrected results,
      in which the architecture narrowly leads (0.272 against 0.265) and the falsification
      condition fixed in advance is not met.
- [ ] Results and Discussion in `manuscript/`.

## Order of work

1. Assemble LUCAS repeat points with their covariate series; confirm how many points
   genuinely repeat, since that number caps everything.
2. Build the blocking scheme first, before any model, so no result is ever produced
   under random splitting by accident.
3. Baselines before the network: persistence, gradient boosting on summary statistics,
   and a process-model run.
4. The full architecture, then the five ablations fixed in the protocol.
6. Write Results and Discussion into `manuscript/`.

Skill on the change in SOC, not on the level, decides the outcome. A model that scores
well on absolute concentration while failing on change has learned persistence.

## Resolved: flooded periods

Flooded ground is invisible to optical sensors as soil, but submergence is itself a
mechanism driving carbon accumulation, so NDWI and MNDWI enter as their own channels
rather than being masked out. Masking would discard the variable that distinguishes a
waterlogged soil from a drained one.


