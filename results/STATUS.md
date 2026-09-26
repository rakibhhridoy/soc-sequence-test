# Where the study stands, 2026-09-25

## Stop here and read this first

A data bug invalidated every neural-network result produced before 2026-09-25. The cause
is fixed and most results are re-run, but **the manuscript still describes the pre-bug
conclusion and must not be sent anywhere until it is rewritten.**

## The bug

Cation exchange capacity is recorded in the LUCAS 2009 release and not in 2015, so it was
missing for every observation whose previous campaign was 2015, which is half the panel.
The per-block mean fill in `build_panel.py` could not fill a column that was absent in its
entirety, and a scaler fitted on a mixed-year fold then produced NaN, which propagated
through the network and returned NaN predictions. Training never improved, early stopping
never triggered, and the reported skill came from partly untrained weights. Gradient
boosting was unaffected, since XGBoost handles missing values natively.

Fixed by dropping CEC, filling residual gaps from the whole panel, and making
`build_panel.build()` raise rather than return a panel containing missing values.

## What the corrected numbers say

Skill against persistence, blocked validation, mineral soils unless stated.

| | Hybrid | Gradient boosting | Persistence |
|---|---|---|---|
| Spatial blocking | 0.276 | 0.265 | 0.000 |
| Forward in time | 0.202 | 0.160 | 0.000 |
| Organic soils (268 points) | 0.446 | 0.434 | 0.000 |

The architecture leads narrowly under every scheme, so **the falsification condition
fixed in advance is not met** and the paper can no longer be written as a refutation.

The substantive finding is in `dynamic_diagnostic.csv`. Replacing the covariate series with
zeros costs only 0.031 skill (0.271 to 0.240), noise costs 0.034, and gradient boosting on
static covariates and the previous carbon value alone reaches 0.239. Static soil properties
and the previous measurement therefore supply roughly 0.24 of the 0.276, the satellite
series add about 0.03, and every method recovers a similar amount. Removing the recurrent
decoder or the convolutional encoder costs nothing measurable; removing the static
embedding costs 0.121.

The defensible headline: the architecture works and modestly outperforms the tabular
baseline, while almost none of its skill comes from the sequence machinery it was built
around.

## Results on disk, all from the corrected panel

- `ablations_mineral.csv` -- main table, tuned at lr 0.001 and dropout 0.2
- `ablations_organic.csv`, `temporal_validation.csv`, `dynamic_diagnostic.csv`
- `tuning.json` -- twelve configurations spanning 0.310 to 0.329

## Unfinished

1. **Main table at the right configuration.** The corrected tuning search picks lr 0.003
   and dropout 0.1, while `ablations_mineral.csv` was produced at lr 0.001 and dropout 0.2.
   A partial run at the correct setting gave 0.272 for the full model against 0.276, so the
   difference looks immaterial, but the table should be regenerated before publication:

       python scripts/run_ablations.py --data data/processed/panel_mineral.npz \
           --epochs 200 --folds 4 --hidden 32 --rnn-hidden 32 --lr 0.003 --dropout 0.1 --ensemble 3

2. **Graph extension**, written and verified on synthetic input but never run on real data:

       python scripts/run_graph.py --data data/processed/panel_mineral.npz --epochs 300 --folds 4
       python scripts/run_graph.py --data data/processed/panel_organic.npz --epochs 300 --folds 3

3. **Manuscript rewrite.** `manuscript/` currently argues the refutation throughout, in the
   title, abstract, results, discussion and conclusions. All of it postdates the bug.

4. **Pre-registration framing.** The preprint was never posted, so it carries no public
   timestamp. The recommendation was to post it and describe the protocol as fixed in
   advance without using the term pre-registered, which no date check can contradict.

## Cautions carried forward

- The external drive disconnected three times during this work. `~/soc_backup` on the
  internal disk holds a copy; refresh it after any substantial run.
- LUCAS may not be redistributed. See `RELEASE.md` before any public release.
- Organic soils carry 16 % optimism from random splitting on 268 points, so that row is
  fragile.

## Update, 2026-09-25 evening

- `ablations_mineral.csv` regenerated at lr 0.003 / dropout 0.1 / ensemble 3. Full model
  0.272, gradient boosting 0.265, no recurrent decoder 0.277, no static embedding 0.149.
  The old table is kept as `ablations_mineral_lr001_do02.csv`.
- Graph extension run (`graph_mineral.csv`, `graph_organic.csv`). **The geographic-graph
  rows are invalid: target leakage.** Every point appears twice (2015 and 2018 targets) at
  identical coordinates, so each node's nearest neighbour is its own twin, and the 2018
  twin's `y_prev` is exactly the 2015 target (verified on all 8,100 pairs). Message passing
  carries it across. Both twins share a spatial block, so blocking does not prevent it.
  The mineral 0.321 must not be reported. Fix before rerunning: drop edges between
  observations of the same point, and keep a directed edge j->i only when times[j] <= times[i].
- Covariate-space graph, mineral: 0.268, no gain over the hybrid. It also selects neighbours
  on features that include `y_prev`, so it needs the same edge rule before it is final.
- Organic graph rows are unstable (CCC near 0, coverage 0.71 to 0.72) and carry the same leak.

## Graph extension rerun with causal edges (2026-09-25, 22:52)

`graphs.causal_edges` now drops same-point edges and any edge from a later round;
`run_graph.py` asserts both. Leaky tables kept as `graph_*_LEAKY.csv`.

| | Geographic | Covariate-space | Hybrid, no graph |
|---|---|---|---|
| Mineral | 0.273 | 0.272 | 0.272 |
| Organic | 0.411 | 0.337 | 0.446 |

Message passing adds nothing on mineral soils; the whole earlier 0.321 was leakage. On
organic soils it is worse than the plain hybrid and the covariate-space row is unstable
(CCC near 0, coverage 0.67). Candidate for the manuscript: one sentence and a supplementary
row, reported as a null result, noting the leak and its fix.

## Manuscript rewritten (2026-09-25, 23:04)

Title, abstract, Introduction close, Results, Discussion, Limitations and Conclusions now
report the corrected numbers: parity-plus under spatial blocking (0.272 vs 0.265), a wider
lead forward in time (0.202 vs 0.160), and sequence components that contribute nothing.
"Pre-registered" replaced by "fixed in advance" throughout. New Table (dynamic diagnostic)
and a graph-extension subsection. Old draft in `manuscript/superseded/`.
Open: organic-soil ablation settings not recorded (lr/dropout unknown); temporal and
dynamic-diagnostic runs are at lr 0.001 / dropout 0.2, disclosed in captions.

## Tuned reruns and input fixes (2026-09-25, ~23:20)

- Organic ablations at lr 0.003 / dropout 0.1 / ensemble 3: full 0.426, GB 0.434, GRU 0.461,
  no decoder 0.365, optimism 18 %. Architecture now just below the baseline on organic soils.
  Previous table (settings unrecorded) kept as `ablations_organic_untracked_settings.csv`.
- Temporal at tuned config: hybrid 0.178 vs GB 0.160 (was 0.202 at lr 0.001 / dropout 0.2,
  kept as `temporal_validation_lr001_do02.csv`). Manuscript reports both as sensitivity.
- Manuscript Input representation, encoder, fusion, ablation text and two figure captions now
  describe the inputs actually used (10 Landsat channels x 60 months; 5 soil properties +
  previous carbon; no climate, management, bulk density or terrain).
- Only remaining run at the pre-search config: dynamic diagnostic (disclosed in its caption).

## Dynamic diagnostic at tuned config (2026-09-25)

Three-member ensembles, lr 0.003 / dropout 0.1: real 0.272 (reproduces Table 1), zeros 0.244,
noise 0.241, GB static+previous 0.239. Series worth ~0.03 as before. Manuscript updated; every
result table is now at the tuned configuration. Old table: `dynamic_diagnostic_lr001_do02_single.csv`.

## GitHub (2026-09-26)

Public repo https://github.com/rakibhhridoy/soc-sequence-test, commit author rakibhhridoy@yahoo.com. manuscript/ is gitignored until the preprint is posted. Manuscript URL placeholder filled. Next: Zenodo (RELEASE.md).

## Zenodo (2026-09-26)

Manual upload of soc-sequence-test.zip. Version DOI 10.5281/zenodo.22973028 (in manuscript), concept DOI 10.5281/zenodo.22973027 (all versions). Record title "SOC Sequence Test".

## Bootstrap (2026-09-26) -- changes the headline

Block bootstrap over 79 spatial blocks, 2,000 replicates (`bootstrap_comparison.csv`).
The main-table gradient boosting was run WITHOUT the previous carbon value, which the hybrid
receives. Given it, boosting matches the hybrid exactly:

| Spatial, mineral | Skill | 95 % CI |
|---|---|---|
| Hybrid, full | 0.272 | 0.247-0.296 |
| GB with previous value | 0.272 | 0.249-0.294 |
| Hybrid minus GB (with previous) | +0.001 | -0.009 to 0.009 |
| Hybrid minus GB (no previous, as in Table 1) | +0.007 | -0.003 to 0.017 |
| No decoder minus full | +0.004 | -0.004 to 0.012 |
| Full minus zero series | +0.028 | 0.019 to 0.037 |

Temporal: hybrid 0.178 vs GB with previous 0.160, difference +0.018 (-0.001 to 0.033, 97 %
of replicates above zero). Under the protocol's wording ("refuted if gradient boosting ...
matches the full architecture"), the fair baseline matches it under spatial blocking.
Manuscript currently says the condition is NOT met and must be revised.

## Climate check, ERA5-Land (2026-09-26)

`climate_comparison.csv`, same folds, paired block bootstrap. Adding monthly temperature,
precipitation and top-layer soil water: hybrid 0.272 -> 0.282 (+0.009, 0.002 to 0.018);
GB with previous 0.272 -> 0.286 (+0.014, 0.006 to 0.023). With climate, hybrid minus GB =
-0.004 (-0.015 to 0.006). Climate helps both a little, helps boosting slightly more, and the
tie stands. Climate ablation table not run.

## Manuscript revised to option 1 (2026-09-26)

Protocol-faithful framing: under spatial blocking the protocol baseline (GB with previous
value) matches the architecture (0.272 each, diff 0.001, CI -0.009 to 0.009), so the
condition fixed in advance is MET. Temporal lead 0.018 reported as probable, not
established. New Methods subsection "Comparing models" (fair baseline, block bootstrap,
ERA5-Land), new bootstrap table, new climate subsection, organic GB-with-previous row
(0.428 vs hybrid 0.426). Previous draft archived in manuscript/superseded/*_parity_2026-09-26.

## EJSS restructure (2026-09-26)

main.tex is now self-contained in EJSS order: Introduction, Materials and methods, Results, Discussion (incl. departures and limitations), Conclusions. Highlights (4, <=100 chars) and 8 alphabetical keywords with no title words. 16 pages, ~4,700 words of prose + ~500 of captions, abstract 1,557 chars. Cell equations and conv/RNN/LSTM/GRU diagrams moved to supplementary.tex (Figs S1-S4). figures_main.tex holds the architecture and validation figures. Old files in manuscript/superseded/. Next: result figures.

## Result figures (2026-09-26)

scripts/make_figures.py -> results/figures/fig_skill, fig_inputs, fig_map (pdf+png), copied to manuscript/figures/. Palette slots 1-2 (blue hybrid, orange GB) validated for CVD. Main text now 18 pages: Fig 3 map of folds (methods), Fig 4 skill + paired differences, Fig 5 input contributions.

## Pending (2026-09-26)

- Author will upload a new Zenodo version later (repo now has bootstrap, climate and figure scripts that the v1 zip lacks). Manuscript DOI left as is by request.

## Soil-science breakdown (2026-09-26)

scripts/soil_breakdown.py (no retraining): skill by land use (cropland 0.280/0.276, grassland 0.254/0.258, ties), by previous-SOC tertile (0.31 / 0.12 / 0.31, U-shape), simple baselines (campaign shift 0.005, campaign x land use 0.027, linear on previous value 0.174, + campaign 0.185 = two thirds of 0.272: mostly regression to the mean). Observed change: cropland median +23% then -11%, grassland -19% both intervals. Map fig_change_map: cell r=0.70, point slope 0.48; overshoot on 82 high-C Irish/W British points (pred -0.52 vs obs -0.16). Manuscript: new Results subsection, Table and Fig; 20 pages.

## Critique fixes in progress (2026-09-26 15:13)

Verified flaws: (1) noise ceiling mixes g/kg and log scales; on the log scale it is 0.215
(median) or negative (rms), below the achieved 0.272, so the construct is invalid and is to be
removed. (2) Texture NaN for all revisited points in 2015/2018 and mean-filled: clay had 3
distinct values in the 2018-target half. FIXED in build_panel (first measured texture carried
forward). (3) Mineral class used max SOC over all rounds incl. the target (selection on the
outcome). FIXED: classed on soc_prev < 120. New panel: 16,462 mineral obs / 8,339 points,
274 organic obs. Also found: early stopping uses a random (not blocked) subset of the training
fold, although the manuscript says blocked.
New scripts/build_panels.py, scripts/robustness.py (5 seeds x fold assignments, linear
mean-reversion baseline, TOST at 0.02 margin chosen post hoc, spatiotemporal validation).
Full rerun queued: results/logs/queue_fixes.sh, per-step logs in results/logs/fixes/.
Old tables in results/tables/superseded_prefix_2026-09-26/, old panels in
data/processed/superseded/2026-09-26/.

## Rerun on corrected panel complete (2026-09-26 18:20)

Spatial (5 replicates pooled): hybrid 0.248, GB fair 0.254, linear mean reversion 0.169;
hybrid - GB -0.006 (95 % -0.022 to 0.009; 90 % inside +/-0.02 -> equivalent at the post hoc
margin); both beat linear by ~0.08; no decoder - hybrid +0.008 (equivalent at 0.02).
Spatiotemporal (3 replicates): hybrid 0.156, GB 0.121, linear 0.140; hybrid - linear +0.016
(-0.020 to 0.051): no method beats mean reversion at new places and later dates.
Same-point temporal (old design): hybrid 0.179 vs GB 0.135, +0.044 (0.017-0.067) -- inflated.
Climate: +0.020 hybrid, +0.015 GB, tie with climate (-0.004). Zeros cost 0.024.
Organic (274 obs, block 276 km): hybrid 0.389, GB no prev 0.363, no decoder 0.396 --
the earlier decoder "reversal" on organic soils is gone. Graph: mineral 0.252/0.251, organic
0.368/0.375, nothing added. Manuscript NOT yet updated.

## Manuscript rewritten on the corrected panel (2026-09-26 evening)

main.tex now reports the rerun: pooled spatial 0.248 vs 0.254 (diff -0.006, -0.022 to 0.009,
equivalent at the post hoc 0.02 margin), mean-reversion floor 0.169, same-point temporal lead
0.044 (0.017-0.067) shown alongside the new-places design (0.035, -0.003 to 0.077; hybrid vs
linear 0.016, -0.020 to 0.051). Noise ceiling removed; Sect. 2.2 now reports the log-scale
correlation of consecutive changes, -0.41 on 8,123 points (old -0.268 was g/kg over all 8,368
points incl. organic). Early stopping disclosed as random 20 % (listed under departures).
Tuning disclosed as run on the earlier panel build. "Previous value decides the comparison"
narrative dropped (GB no-prev now 0.250 = hybrid). Organic "reversal" and GRU claim dropped.
New numbers computed for the text: variogram 276 km / 92 blocks / median 104 obs; TN vs log
prev C r = 0.61; map cells r = 0.72 (357 cells); Atlantic west (lon < -5, lat > 51) 96 obs,
pred -0.33 vs obs -0.05; all obs > 40 g/kg pred -0.76 vs obs -0.74. Organic GB with previous
value computed ad hoc on the same folds: 0.378, results/tables/organic_boosting_with_previous.csv.
Previous draft: manuscript/superseded/*_prefix_2026-09-26.
