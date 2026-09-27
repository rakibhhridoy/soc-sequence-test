"""Do the dynamic covariates contribute anything at all?

Three ablations of Table 2 returned identical skill, which is what appears when the
component removed was doing no work. This replaces the covariate series with zeros and
with noise, holding the architecture fixed. If skill is unchanged, the sequence pathway is
inert and the model is running on static soil properties and the previous carbon value.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, tuning, baselines, evaluate, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402


def main() -> int:
    base = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    config.seed_everything()
    device = config.device()
    folds, _, groups = train.spatial_folds(base, n_folds=4)
    # configurations selected for the full model on the real series, held fixed here
    fc = tuning.fold_configs(train.with_previous(base), folds, groups, device, tag="mineral spatial")
    cfg = train.TrainConfig(epochs=200)
    rng = np.random.default_rng(0)

    rows = []
    for name, maker in (
        ("real covariate series", lambda d: d.x_dyn),
        ("series replaced by zeros", lambda d: np.zeros_like(d.x_dyn)),
        ("series replaced by noise", lambda d: rng.normal(size=d.x_dyn.shape).astype(np.float32)),
    ):
        d = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
        d.x_dyn = maker(d)
        out = train.cross_validate(train.with_previous(d), cfg, folds, device, n_ensemble=3,
                                   groups=groups, fold_cfgs=fc)
        rows.append(dict(model=f"hybrid, {name}", **_flat(out["metrics"])))
        print(f"  {name:28s} skill {rows[-1]['skill']:+.3f}")

    # tabular reference using no dynamic information whatsoever
    feats = np.hstack([base.x_static, base.y_prev[:, None]])
    mu = np.full(len(base.y), np.nan)
    for tr, te in folds:
        mu[te] = baselines.fit_gradient_boosting(feats[tr], base.y[tr]).predict(feats[te])
    ok = ~np.isnan(mu)
    rows.append(dict(model="gradient boosting, static + previous only",
                     **_flat(evaluate.evaluate_level_and_change(base.y_prev[ok], base.y[ok], mu[ok]))))
    print(f"  {'GBM static + previous only':28s} skill {rows[-1]['skill']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "dynamic_diagnostic.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def _flat(res):
    return {"rmse": res["level"]["rmse"], "ccc": res["level"]["ccc"],
            "skill": res["skill_vs_persistence"], "n": res["level"]["n"]}


if __name__ == "__main__":
    raise SystemExit(main())
