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
