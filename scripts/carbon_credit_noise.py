"""What the measurement error of a LUCAS value means for soil carbon crediting.

The error variance var(e) of a single log-carbon value, identified in point_change.py from
three visits to 8,221 points, is translated into the terms of carbon crediting under the EU
carbon removals and carbon farming certification (Regulation (EU) 2024/3012 and its carbon
farming methodologies, Commission Delegated Regulation annex C(2026) 4666). Five quantities,
each per hectare over one six-year measurement cycle, the interval of the Soil Monitoring Law:

1. Apparent stock change from noise. With no true change, a remeasured point shows a change
   of exp(eps) - 1 in concentration, eps ~ N(0, 2 var(e)). Its mean absolute size is set
   against the gain that the 4 per 1000 target implies over six years, 1.004^6 - 1.
2. Detectable change and sample size. The standard deviation D of the measured six-year
   change in log carbon, which holds noise and true heterogeneity, gives the number of
   resampled locations needed to detect the 4 per 1000 gain, both with the methodology's
   equation (8), n >= z^2 D^2 / E^2 at z = 0.52 (one-sided 70 %) and E the gain multiplied by
   the confidence level, the tolerable error its text gives as an example (E equal to the
   gain is reported beside it), and with a
   conventional one-sided 5 % test at 80 % power, and the minimum detectable change for a
   given number of locations.
3. Crediting without change. The methodology deducts the uncertainty factor
   UNC = max(8 %, t70 SE / mean change) (equation 12), so a project is credited mean change
   minus t70 SE. With no true change the mean change is pure noise, and a project is
   credited with probability 1 - Phi(0.52) = 0.30, with an expected credit of
   SE [phi(0.52) - 0.52 (1 - Phi(0.52))]. With the sample size of equation (8), SE = E / 0.52,
   and the expected credit becomes a fixed share of the 4 per 1000 gain.
4. Reversion credited as sequestration. Fields enrolled because their baseline value is low
   show a later gain from reversion towards the mean alone. The 2015 value carries the error
   that the 2015-2018 change shares, the 2009 value does not, and the instrumented slope of
   change on level is close to zero, so the difference between the mean 2015-2018 change in
   the lowest third of points by their 2015 value and by their 2009 value is the
   reversion gain over one remeasurement. It is set by the error, not by the interval, so it
   does not grow with the length of the cycle.
5. The model route. The methodology accepts a model whose R^2 on observed change, equation (7),
   exceeds zero. That R^2 is computed for the mean-reversion floor, which forecasts no change
   beyond reversion, and for the hybrid and boosting, on the five spatial replicates of
   robustness.py.

Stock is concentration x fine-earth bulk density x depth, 0-20 cm as sampled by LUCAS,
without coarse fragments. Bulk density is the LUCAS fine-earth mean by land cover of Chen et
al. (2024, ESSD 16, 2367), 1.16, 1.00 and 1.02 g cm-3 for cropland, grassland and other
cover, with the measured cropland mean of Panagos et al. (2024, AGEE 364, 108907), 1.26, as
an upper case. Values scale linearly with bulk density and depth. Carbon is converted to CO2
by 44/12 and valued at the 2024 average price of agriculture credits in the voluntary market,
US$7.66 t-1 (Ecosystem Marketplace 2025, Table 3), and at the 2025 average EU allowance
price, EUR 74 t-1 (ESMA), an upper reference that CRCF units do not trade at.

Intervals resample the spatial blocks of the main analysis.

    python scripts/carbon_credit_noise.py
"""
from __future__ import annotations
import sys, pathlib, os
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import config, train  # noqa: E402
from run_ablations import load_npz  # noqa: E402
from bootstrap_comparison import PRED_DIR  # noqa: E402
from robustness import R_SPATIAL  # noqa: E402

N_BOOT = int(os.environ.get("N_BOOT", 2000))
DEPTH_CM = 20
BD = {"fine earth by land cover (Chen et al. 2024)": {"B": 1.16, "E": 1.00, "other": 1.02},
      "measured cropland mean (Panagos et al. 2024)": {"B": 1.26, "E": 1.26, "other": 1.26}}
CO2_PER_C = 44 / 12
PRICES = {"voluntary agriculture, US$ 7.66": 7.66, "EU allowance 2025, EUR 74": 74.0}
GAIN_4P1000 = 1.004 ** 6 - 1                # relative gain over six years at 0.4 % per year
Z70, Z95, Z80 = norm.ppf(0.70), norm.ppf(0.95), norm.ppf(0.80)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)   # E[f(Z)], Z ~ N(0, 1)


def stock_t_c(conc_gkg, bd):
    """Organic carbon stock, t C per hectare, 0-20 cm: g/kg x g/cm3 x cm x 0.1."""
    return conc_gkg * bd * DEPTH_CM * 0.1


def mean_abs_rel_change(sd_log):
    """E|exp(eps) - 1| for eps ~ N(0, sd_log^2)."""
    return float(np.sum(GH_W * np.abs(np.expm1(sd_log * GH_X))) / np.sqrt(2 * np.pi))


def quantities(x09, x15, x18, stock):
    """Everything reported, for one set of points (log g/kg values, stock in t C/ha)."""
    d1, d2 = x15 - x09, x18 - x15
    var_e = -np.cov(d1, d2)[0, 1]
    s_bar = np.median(stock)               # a typical point; the mean is pulled up by a long tail
    out = {"var_e": var_e, "median stock, t C/ha": s_bar, "mean stock, t C/ha": np.mean(stock)}

    # 1. apparent change from noise against the 4 per 1000 gain
    noise_sd = np.sqrt(2 * max(var_e, 0))
    rel = mean_abs_rel_change(noise_sd)
    out["noise SD of six-year change, log"] = noise_sd
    out["mean absolute apparent change from noise, % of stock"] = 100 * rel
    out["mean absolute apparent change from noise, t CO2/ha"] = rel * s_bar * CO2_PER_C
    out["4 per 1000 gain over six years, t CO2/ha"] = GAIN_4P1000 * s_bar * CO2_PER_C
    out["noise / 4 per 1000 gain"] = rel / GAIN_4P1000

    # 2. detectability: D holds noise and true heterogeneity of change
    D = np.std(d1, ddof=1)
    delta = np.log1p(GAIN_4P1000)
    out["SD of measured six-year change, log"] = D
    out["locations to detect 4 per 1000, CRCF eq. 8, E = 0.7 x gain"] = (Z70 * D / (0.7 * delta)) ** 2
    out["locations to detect 4 per 1000, CRCF eq. 8, E = gain"] = (Z70 * D / delta) ** 2
    out["locations to detect 4 per 1000, 5 % test, 80 % power"] = ((Z95 + Z80) * D / delta) ** 2
    out["locations to detect 4 per 1000, noise only, 5 % test, 80 % power"] = \
        ((Z95 + Z80) * noise_sd / delta) ** 2
    for n in (1, 10, 100):
        mdc = np.expm1((Z95 + Z80) * D / np.sqrt(n))
        out[f"minimum detectable six-year change, {n} locations, %"] = 100 * mdc
        out[f"minimum detectable six-year change, {n} locations, t CO2/ha"] = \
            mdc * s_bar * CO2_PER_C

    # 3. crediting with no true change, sample size of eq. 8 (SE = E / z70, E = 0.7 x gain)
    share = (norm.pdf(Z70) - Z70 * norm.sf(Z70)) / Z70 * 0.7   # E[(Z - z70)+] SE / gain
    out["probability a no-change project is credited"] = norm.sf(Z70)
    out["expected credit without change, share of 4 per 1000 gain"] = share
    out["expected credit without change, t CO2/ha"] = share * delta * s_bar * CO2_PER_C

    # 4. reversion credited as sequestration, lowest third by baseline
    low15 = x15 <= np.percentile(x15, 100 / 3)
    low09 = x09 <= np.percentile(x09, 100 / 3)
    rev = np.mean(d2[low15]) - np.mean(d2[low09])
    out["reversion gain, lowest third by baseline, log"] = rev
    out["reversion gain, lowest third by baseline, %"] = 100 * np.expm1(rev)
    out["reversion gain, lowest third by baseline, t CO2/ha"] = \
        np.expm1(rev) * np.median(stock[low15]) * CO2_PER_C
    out["reversion gain / 4 per 1000 gain"] = np.expm1(rev) / GAIN_4P1000
    return out


def crcf_r2(obs, pred):
    """Equation (7) of the methodology, on change: 1 - SS(P - O) / SS(O - mean O)."""
    ok = ~np.isnan(pred)
    o, p = obs[ok], pred[ok]
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


def main() -> int:
    data = load_npz(config.DATA_PROCESSED / "panel_mineral.npz")
    y, y0, year, pid = data.y, data.y_prev, data.times, data.point_id.astype(str)
    _, _, groups = train.spatial_folds(data, n_folds=4)
    a = pd.DataFrame(dict(pid=pid, year=year, y=y, y0=y0, g=groups))
    w = a.pivot(index="pid", columns="year").dropna(subset=[("y", 2015.0), ("y", 2018.0)])
    x09, x15, x18 = (w[("y0", 2015.0)].to_numpy(), w[("y", 2015.0)].to_numpy(),
                     w[("y", 2018.0)].to_numpy())
    g_pt = w[("g", 2015.0)].to_numpy()
    p = pd.read_parquet(config.DATA_PROCESSED / "lucas_repeat_panel.parquet",
                        columns=["point_id", "year", "land_cover"])
    p["point_id"] = p.point_id.astype(str)
    lc = (p[p.year == 2015].set_index("point_id").land_cover.reindex(w.index)
          .fillna("").str[:1].to_numpy())
    lc = np.where(np.isin(lc, ["B", "E"]), lc, "other")
    print(f"{len(w):,} points; land cover 2015 {pd.Series(lc).value_counts().to_dict()}")
    conc = np.exp(x15)                       # the reference level is the 2015 value

    rng = np.random.default_rng(0)
    blocks = np.unique(g_pt)
    members = {b: np.where(g_pt == b)[0] for b in blocks}
    rows = []
    for scen, bd in BD.items():
        stock = stock_t_c(conc, np.vectorize(bd.get)(lc))
        est = quantities(x09, x15, x18, stock)
        draws = []
        for _ in range(N_BOOT):
            idx = np.concatenate([members[b] for b in rng.choice(blocks, len(blocks))])
            draws.append(quantities(x09[idx], x15[idx], x18[idx], stock[idx]))
        for k, v in est.items():
            vals = np.array([d[k] for d in draws])
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            rows.append(dict(bulk_density=scen, quantity=k, estimate=v, ci95_low=lo,
                             ci95_high=hi))
        for k in [q for q in est if q.endswith("t CO2/ha")]:
            for pname, price in PRICES.items():
                vals = np.array([d[k] for d in draws]) * price
                lo, hi = np.nanpercentile(vals, [2.5, 97.5])
                rows.append(dict(bulk_density=scen, quantity=k.replace("t CO2/ha", "")
                                 + f"value per ha at {pname}", estimate=est[k] * price,
                                 ci95_low=lo, ci95_high=hi))

    # 5. the model route: equation (7) R^2 on observed change, five spatial replicates
    obs = y - y0
    for name in ("linear", "boosting", "hybrid"):
        r2 = [crcf_r2(obs, np.load(PRED_DIR / f"robustness_rep{r}.npz")[name] - y0)
              for r in range(R_SPATIAL)]
        label = {"linear": "mean-reversion floor"}.get(name, name)
        rows.append(dict(bulk_density="not used", quantity=f"CRCF eq. 7 R2 on change, {label}",
                         estimate=np.mean(r2), ci95_low=np.min(r2), ci95_high=np.max(r2)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    out = config.TABLES / "carbon_credit_noise.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  (the R2 rows give the range over replicates, not an interval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
