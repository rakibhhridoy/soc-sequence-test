"""Launch Earth Engine exports of the monthly covariate series for the LUCAS points.

One task per calendar year, each carrying twelve months of nine variables for every
point. The five-year window for a survey round is assembled locally afterwards, so the
years shared between rounds are exported once rather than three times.

Results land as CSVs in Google Drive (folder `soc_article`) and must be downloaded into
data/raw/covariates/ before assembly.

    python scripts/export_covariates.py            # launch every missing year
    python scripts/export_covariates.py --status   # report on running tasks
"""
from __future__ import annotations
import argparse, sys, pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, covariates  # noqa: E402

ROUNDS = (2009, 2015, 2018)
POINTS = config.DATA_PROCESSED / "lucas_points.csv"


def needed_years() -> list[int]:
    yrs = set()
    for r in ROUNDS:
        yrs.update(covariates.window_years(r, 5))
    return sorted(yrs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="years already exported")
    args = ap.parse_args()

    ee = covariates.init()
    if args.status:
        for t in ee.data.getTaskList()[:25]:
            if t["description"].startswith("lucas_cov"):
                print(f"{t['description']:24s} {t['state']:10s} {t.get('error_message','')}")
        return 0

    pts = pd.read_csv(POINTS, dtype={"point_id": str})
    years = [y for y in needed_years() if y not in args.skip]
    print(f"{len(pts):,} points | launching {len(years)} tasks: {years}")
    for y in years:
        t = covariates.export_year(pts, y)
        print(f"  {y}: task {t.id}")
    print("\nWatch progress at https://code.earthengine.google.com/tasks or with --status")
    print("Then download the Drive folder 'soc_article' into data/raw/covariates/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
