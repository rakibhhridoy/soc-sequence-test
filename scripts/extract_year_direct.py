"""Fetch one calendar year of covariates without the Earth Engine batch queue.

The batch exporter is far faster when the queue is moving, but a task can sit in READY
indefinitely. This path samples points directly in chunks and writes the same wide CSV
the exporter produces, so the rest of the pipeline cannot tell the difference.

    python scripts/extract_year_direct.py 2010
"""
from __future__ import annotations
import argparse, sys, pathlib, time
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, covariates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    pts = pd.read_csv(config.DATA_PROCESSED / "lucas_points.csv", dtype={"point_id": str})
    out_path = config.DATA_RAW / "covariates" / f"lucas_covariates_{args.year}.csv"
    part_dir = config.DATA_INTERIM / f"cov_{args.year}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    chunks = [pts.iloc[i:i + args.chunk] for i in range(0, len(pts), args.chunk)]
    print(f"{len(pts):,} points in {len(chunks)} chunks of {args.chunk}")
    frames = []
    for i, ch in enumerate(chunks):
        part = part_dir / f"part_{i:03d}.csv"
        if part.exists():                       # resume after an interruption
            frames.append(pd.read_csv(part, dtype={"point_id": str}))
            print(f"  chunk {i+1}/{len(chunks)}: cached")
            continue
        t0 = time.time()
        long = covariates.monthly_series(ch, end_date=f"{args.year + 1}-01-01",
                                         n_months=12)
        wide = long.pivot(index="point_id", columns="month",
                          values=covariates.COMMON + covariates.INDICES)
        wide.columns = [f"{b}_{m.replace('-', '')}" for b, m in wide.columns]
        wide = wide.reset_index()
        wide.to_csv(part, index=False)
        frames.append(wide)
        print(f"  chunk {i+1}/{len(chunks)}: {len(wide)} rows, {time.time()-t0:.0f}s")

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({full.shape[0]} rows x {full.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
