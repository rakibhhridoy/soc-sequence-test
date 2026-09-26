"""Build the model-ready panels from the LUCAS rounds and the exported covariates.

Writes `panel.npz` (all observations) and its mineral and organic subsets, classed by the
previous carbon value so that no observation is selected on its own target. Earlier panels
are moved to `data/processed/superseded/` rather than overwritten.

    python scripts/build_panels.py
"""
from __future__ import annotations
import sys, pathlib, shutil, datetime
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import config, build_panel  # noqa: E402


def main() -> int:
    out = build_panel.build()
    proc = config.DATA_PROCESSED
    old = proc / "superseded" / datetime.date.today().isoformat()
    old.mkdir(parents=True, exist_ok=True)
    for f in proc.glob("panel*.npz"):
        shutil.move(str(f), old / f.name)
    np.savez(proc / "panel.npz", **out)
    m = out["mineral"]
    for name, mask in (("panel_mineral", m), ("panel_organic", ~m)):
        np.savez(proc / f"{name}.npz", **{k: v[mask] for k, v in out.items()})
        print(f"{name}: {int(mask.sum()):,} observations, "
              f"{len(np.unique(out['point_id'][mask])):,} points")
    s = out["x_static"][m]
    print("distinct clay values by target year:",
          {int(y): len(np.unique(s[out['times'][m] == y, 0])) for y in np.unique(out["times"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
