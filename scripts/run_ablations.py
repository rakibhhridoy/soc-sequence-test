"""Five ablations of the hybrid, plus baselines, under blocked validation.

Each ablation is a subtraction from the full model, not a separately tuned alternative,
so a difference in skill is attributable to the component removed. Every run uses the
configuration selected for the full model inside each training fold (src/tuning.py).

    1  no recurrent decoder      does temporal state add anything?
    2  no learned encoder        does the CNN beat hand-engineered yearly summaries?
    3  GRU in place of LSTM      the controlled cell comparison
    4  no static embedding       do soil properties and the previous value add anything?
    5  no previous observation   how much rests on the last measured carbon value?

Baselines: persistence, and gradient boosting on summary statistics.
The random k-fold run is reported as the optimism reference, never as a result.

Run on synthetic data (default) to check the machinery:
    python scripts/run_ablations.py --synthetic
Run on the real panel once it exists:
    python scripts/run_ablations.py --data data/processed/panel.npz
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, tuning, blocking, evaluate, baselines, train  # noqa: E402


def make_synthetic(n_points=600, n_times=3, C=9, T=60, P=5, seed=0) -> train.Dataset:
    """A panel with a known structure, to verify the machinery end to end.

    SOC responds to a recent-window mean of one dynamic channel and to one static
    covariate, and carries strong persistence, so persistence scores well on the level
    and the model has to beat it on change. Points are clustered in space, so blocked
    validation is genuinely harder than random splitting.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 200_000, size=(40, 2))
    coords_pt = centres[rng.integers(0, len(centres), n_points)] + rng.normal(0, 3_000, (n_points, 2))
    static_pt = rng.normal(size=(n_points, P))
    soc0 = 20 + 4 * static_pt[:, 0] + rng.normal(0, 2, n_points)

    x_dyn, x_static, y, y_prev, coords, times, pid = [], [], [], [], [], [], []
    soc_prev = soc0.copy()
    for t in range(n_times):
        xd = rng.normal(size=(n_points, C, T))
        xd[:, 0, :] += np.sin(np.linspace(0, 6 * np.pi, T))[None, :]
        drive = xd[:, 0, -24:].mean(1) + 0.5 * xd[:, 1, :].mean(1)
        soc = soc_prev + 0.8 * drive + 0.3 * static_pt[:, 1] + rng.normal(0, 0.4, n_points)
        x_dyn.append(xd); x_static.append(static_pt); y.append(soc); y_prev.append(soc_prev)
        coords.append(coords_pt); times.append(np.full(n_points, 2009 + 3 * t, float))
        pid.append(np.arange(n_points))
        soc_prev = soc

    return train.Dataset(
        x_dyn=np.concatenate(x_dyn).astype(np.float32),
        x_static=np.concatenate(x_static).astype(np.float32),
        y=np.concatenate(y).astype(np.float32),
        y_prev=np.concatenate(y_prev).astype(np.float32),
        coords=np.concatenate(coords), times=np.concatenate(times),
        point_id=np.concatenate(pid),
        delta_max=np.full(n_points * n_times, 6.0, np.float32),
        x_dyn_next=np.concatenate(x_dyn).astype(np.float32))


def load_npz(path) -> train.Dataset:
    """Load a panel, keeping only the fields the Dataset defines.

    The saved file also carries untransformed carbon and the mineral flag, which are
    useful for reporting but are not model inputs.
    """
    import dataclasses
    z = np.load(path, allow_pickle=True)
    fields = {f.name for f in dataclasses.fields(train.Dataset)}
    return train.Dataset(**{k: z[k] for k in z.files if k in fields})


def run_baselines(data: train.Dataset, folds) -> list[dict]:
    rows = []
    res = evaluate.evaluate_level_and_change(data.y_prev, data.y, baselines.persistence(data.y_prev))
    rows.append(dict(model="persistence", **_flat(res)))

    feats = np.hstack([baselines.summary_features(data.x_dyn), data.x_static])
    mu = np.full(len(data), np.nan)
    for tr, te in folds:
        gbm = baselines.fit_gradient_boosting(feats[tr], data.y[tr])
        mu[te] = gbm.predict(feats[te])
    ok = ~np.isnan(mu)
    res = evaluate.evaluate_level_and_change(data.y_prev[ok], data.y[ok], mu[ok])
    rows.append(dict(model="gradient boosting (summary stats)", **_flat(res)))
    return rows


def _flat(res: dict) -> dict:
    return {"rmse_level": res["level"]["rmse"], "ccc_level": res["level"]["ccc"],
            "rmse_change": res["change"]["rmse"], "ccc_change": res["change"]["ccc"],
            "rmse_change_persistence": res["persistence_change_rmse"],
            "skill_vs_persistence": res.get("skill_vs_persistence", np.nan),
            "coverage_90": res.get("coverage_90", np.nan), "n": res["level"]["n"]}


# Each ablation subtracts one component from the full model. All run at the configuration
# selected for the full model inside each training fold, never tuned separately, so a
# difference reflects the component and not its hyperparameters.
ABLATIONS = [
    ("full model (LSTM)",            dict()),
    ("5. no previous observation",   dict(no_prev=True)),
    ("1. no recurrent decoder",      dict(use_recurrent=False)),
    ("2. no learned encoder",        dict(use_encoder=False)),
    ("3. GRU in place of LSTM",      dict(cell="gru")),
    ("4. no static embedding",       dict(use_static=False)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--block-size", type=float, default=None,
                    help="metres; default fits the SOC variogram, which is unstable on small panels")
    args = ap.parse_args()

    if args.data:
        data = load_npz(args.data)
        print(f"loaded {args.data}: {len(data)} observations")
    elif args.synthetic:
        data = make_synthetic()
        print(f"synthetic panel: {len(data)} observations, {data.point_id.max()+1} points")
    else:
        print("give --data or --synthetic"); return 1

    config.seed_everything()
    device = config.device()
    folds, block_size, groups = train.spatial_folds(data, n_folds=args.folds,
                                                    block_size=args.block_size)
    print(f"device={device.type} | spatial block size={block_size:,.0f} m | folds={len(folds)}\n")

    stem = "synthetic" if args.synthetic else pathlib.Path(args.data).stem.replace("panel_", "")
    data_prev = train.with_previous(data)          # previous observation as an input
    fold_cfgs = tuning.fold_configs(data_prev, folds, groups, device, tag=f"{stem} spatial")
    rows = run_baselines(data, folds)
    for name, kw in ABLATIONS:
        kw = dict(kw)
        no_prev = kw.pop("no_prev", False)
        cfg = train.TrainConfig(epochs=args.epochs, **kw)
        t0 = time.time()
        d = data if no_prev else data_prev
        out = train.cross_validate(d, cfg, folds, device, n_ensemble=args.ensemble,
                                   groups=groups, fold_cfgs=fold_cfgs)
        rows.append(dict(model=name, **_flat(out["metrics"])))
        print(f"  {name:32s} rmse={rows[-1]['rmse_level']:.3f}  "
              f"skill_vs_persistence={rows[-1]['skill_vs_persistence']:+.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # optimism reference: the same full model under random splitting, with its own nested
    # search and random early stopping, so nothing about it is blocked
    rnd = list(blocking.random_kfold(len(data), k=args.folds, seed=0))
    rnd_cfgs = tuning.fold_configs(data_prev, rnd, None, device, tag=f"{stem} random")
    out = train.cross_validate(data_prev, train.TrainConfig(epochs=args.epochs), rnd, device,
                               n_ensemble=args.ensemble, fold_cfgs=rnd_cfgs)
    rows.append(dict(model="full model, RANDOM k-fold (optimism reference)", **_flat(out["metrics"])))

    df = pd.DataFrame(rows)
    cols = ["model", "rmse_level", "ccc_level", "skill_vs_persistence",
            "rmse_change_persistence", "coverage_90", "n"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    config.TABLES.mkdir(parents=True, exist_ok=True)
    # named after the input, so one run cannot overwrite another's results
    out_path = config.TABLES / f"ablations_{stem}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    blocked = df.loc[df.model == "full model (LSTM)", "rmse_level"].iloc[0]
    random_ = df.loc[df.model.str.contains("RANDOM"), "rmse_level"].iloc[0]
    print(f"\noptimism from random splitting: rmse {random_:.3f} vs {blocked:.3f} blocked "
          f"({100*(blocked-random_)/blocked:+.0f} %)")
    print("note: RMSE on change equals RMSE on level by construction; skill_vs_persistence "
          "is the column that decides whether a model beats assuming no change.")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
