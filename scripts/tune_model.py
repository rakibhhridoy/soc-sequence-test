"""Summarise the nested hyperparameter selection.

The search itself runs inside every training fold (``src/tuning.py``), triggered by the
analysis scripts and cached in ``results/tables/nested_tuning.json``. This writes one row
per fold with the configuration chosen, its inner-split skill and the spread of the grid,
which is what the manuscript reports.

    python scripts/tune_model.py
"""
from __future__ import annotations
import sys, pathlib, json
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, tuning  # noqa: E402


def main() -> int:
    if not tuning.CACHE.exists():
        print("no nested selections cached yet; run the analysis scripts first")
        return 1
    cache = json.loads(tuning.CACHE.read_text())
    rows = []
    for v in cache.values():
        skills = [g["skill"] for g in v["grid"]]
        b = v["best"]
        rows.append(dict(tag=v["tag"], n_train=v["n_train"], n_inner_val=v["n_inner_val"],
                         hidden=b["hidden"], rnn_hidden=b["rnn_hidden"], lr=b["lr"],
                         dropout=b["dropout"], best_skill=b["skill"],
                         grid_min=min(skills), grid_max=max(skills)))
    df = pd.DataFrame(rows).sort_values("tag")
    out = config.TABLES / "nested_tuning_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
