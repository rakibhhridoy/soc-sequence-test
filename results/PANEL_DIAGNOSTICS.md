# LUCAS panel diagnostics (2026-09-23)

`data/processed/lucas_repeat_panel.parquet`: 8,368 cropland and grassland points, each
measured in 2009, 2015 and 2018. 25,104 observations, no gaps.

| Round | mean SOC (g/kg) | median | sd |
|---|---|---|---|
| 2009 | 22.66 | 15.60 | 33.42 |
| 2015 | 25.70 | 17.64 | 33.45 |
| 2018 | 20.22 | 15.33 | 27.16 |

## Three findings that set the terms for modelling

**Part of the measured change is noise.** Change over 2009 to 2015 correlates with change
over 2015 to 2018 at -0.268. A point that rises then falls, more often than chance, is
the signature of mean reversion rather than of a trend, so some of the apparent change is
measurement and re-sampling variability. LUCAS re-samples near a point, not the identical
spot, which is enough to produce this.

**A 3 % tail dominates the spread.** Organic soils above 120 g/kg are 268 points, 3.2 % of
the panel, and removing them cuts the standard deviation of change from 40.2 to 15.8 g/kg.
Mineral and organic soils are therefore analysed separately, with mineral soils carrying
the main result.

**The target needs a log transform.** Skewness of 8.5 falls to 0.57 on a log scale. Median
change on mineral soils is -6.4 % over the nine years.

## The noise floor, and what skill is attainable

Within-point standard deviation across the three rounds is 5.78 g/kg at the median for
mineral soils, about 36 % of the median level of 15.94 g/kg. That figure is an upper bound
on the noise, since it also contains whatever real change occurred.

Treating it as noise, the error on a difference is roughly 5.78 x sqrt(2) = 8.2 g/kg
against a persistence RMSE of 15.8 g/kg. The highest skill score any model could reach is
therefore about 0.48, and a result near that value would mean the method had extracted
essentially all the recoverable signal. Anything reported against a ceiling of 1.0 would
misstate what the data can support.

The replicate-measurement dataset on ESDAC (inter- and intra-laboratory replicates for
LUCAS carbon) would separate laboratory error from field re-sampling error and tighten
this bound. Worth requesting.

## Decisions taken

1. Target is log SOC concentration; predictions are back-transformed for reporting.
2. Mineral soils (< 120 g/kg in every round) are the primary analysis; organic soils are
   reported separately and never pooled silently.
3. Skill is judged against the noise ceiling of roughly 0.48, not against 1.0.
