"""Count the LUCAS points that repeat across campaigns.

This number caps what any model can be shown to do, so it is established and recorded
before a model is written.

Run: python scripts/lucas_repeat_count.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, data_lucas  # noqa: E402

RAW = config.DATA_RAW / "lucas"


def main() -> int:
    if not RAW.exists():
        print(f"No LUCAS data yet at {RAW}\nSee data/LUCAS_HOWTO.md for how to obtain it.")
        return 1
    df = data_lucas.load_all(RAW)
    print(f"loaded {len(df):,} rows across campaigns {sorted(df.year.unique())}\n")

    panel, summary = data_lucas.repeat_panel(df)
    print(summary.to_string(index=False))
    print(f"\nusable repeat points: {panel.point_id.nunique():,}")

    config.TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES / "lucas_repeat_counts.csv", index=False)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.DATA_PROCESSED / "lucas_repeat_panel.parquet", index=False)
    print(f"\nwrote {config.TABLES / 'lucas_repeat_counts.csv'}")
    print(f"wrote {config.DATA_PROCESSED / 'lucas_repeat_panel.parquet'}")

    n = panel.point_id.nunique()
    if n < 500:
        print("\nFewer than 500 repeat points. That is thin for a sequence model, and the "
              "plan should be revisited before training rather than after.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
