"""Part 2: same-index evidence, fund-level opening noise and risk tests.

Funds from three families (iShares, Vanguard, SPDR) that track the SAME style
index have the same efficient return, so any difference between their session
returns is pricing error (plus negligible tracking difference).

Outputs (tables/, results/, figures/):
  tab_funds      appendix list of funds with liquidity and print-quality data
  tab_samecorr   same-index sister correlations by session
  tab_same       own vs same-index cross reversal, IV, noise variance (main test)
  tab_samesub    same-index own/cross correlations by subperiod
  tab_fundnoise  fund-level opening and closing noise (three-cornered hat)
  tab_fundliq    fund-level noise vs liquidity with index-by-year fixed effects
  tab_samequal   reversal conditional on print quality, same-index sample
  tab_samerisk   noise-corrected session risk of the 13 spreads
  tab_forecast   overnight variance forecasts: raw vs corrected (robust-loss test)
  tab_var        overnight VaR backtests: raw vs corrected
  tab_sameprof   gross intraday reversal returns: own vs same-index sister signal
  tab_pairs      appendix: all 22 ordered same-index pairs
  fig_fundnoise, fig_samerolling, fig_forecast
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, TAB, FIG = ROOT / "data", ROOT / "results", ROOT / "tables", ROOT / "figures"
START, END = "2011-01-03", "2026-09-25"
LAGS, BP, WIN = 10, 1e4, 250
FAM = {"I": "iShares", "V": "Vanguard", "S": "SPDR"}

# index -> family -> (growth, value)
STYLE = {
    "Russell 1000": {"I": ("IWF", "IWD"), "V": ("VONG", "VONV")},
    "Russell 2000": {"I": ("IWO", "IWN"), "V": ("VTWG", "VTWV")},
    "S\\&P 500": {"I": ("IVW", "IVE"), "V": ("VOOG", "VOOV"), "S": ("SPYG", "SPYV")},
    "S\\&P 400": {"I": ("IJK", "IJJ"), "V": ("IVOG", "IVOV"), "S": ("MDYG", "MDYV")},
    "S\\&P 600": {"I": ("IJT", "IJS"), "V": ("VIOG", "VIOV"), "S": ("SLYG", "SLYV")},
}
BENCH = {
    "Russell 1000": {"I": "IWB", "V": "VONE"},
    "Russell 2000": {"I": "IWM", "V": "VTWO"},
    "S\\&P 500": {"I": "IVV", "V": "VOO", "S": "SPY"},
    "S\\&P 400": {"I": "IJH", "V": "IVOO", "S": "MDY"},
    "S\\&P 600": {"I": "IJR", "V": "VIOO"},
}
INDEXNAME = {"Russell 1000": "Russell 1000", "Russell 2000": "Russell 2000", "S\\&P 500": "S\\&P 500",
             "S\\&P 400": "S\\&P MidCap 400", "S\\&P 600": "S\\&P SmallCap 600"}


# ------------------------------------------------------------------ data
def load(t: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{t}.csv", index_col=0, parse_dates=True).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d[(d.Open > 0) & (d.Close > 0) & (d["Adj Close"] > 0)]
    r = pd.DataFrame(index=d.index)
    r["cc"] = d["Adj Close"].pct_change()
    r["id"] = d.Close / d.Open - 1
    r["on"] = (1 + r.cc) / (1 + r.id) - 1
    r["stale"] = (d.Open == d.Close.shift()).astype(float)
    r["ext"] = ((d.Open >= d.High) | (d.Open <= d.Low)).astype(float)
    r["dvol"] = d.Volume * d.Close
    r["relvol"] = d.Volume / d.Volume.rolling(60, min_periods=20).median().shift()
    r["first"] = d.index[0]
    return r


TICK = sorted({t for g in STYLE.values() for p in g.values() for t in p} | {t for g in BENCH.values() for t in g.values()})
RAW = {t: load(t) for t in TICK}
IDX = None
for t in TICK:
    IDX = RAW[t].index if IDX is None else IDX.intersection(RAW[t].index)
IDX = IDX[(IDX >= START) & (IDX <= END)]
# 24 August 2015: market-wide ETF opening dislocation (SEC 2015). A single day that
# dominates second moments of the smallest funds; excluded here, included in robustness.
EXCL = pd.Timestamp("2015-08-24")
IDX_ALL = IDX
IDX = IDX.drop(EXCL)
R_ALL = {t: RAW[t].reindex(IDX_ALL) for t in TICK}
R = {t: RAW[t].reindex(IDX) for t in TICK}
YEARS = IDX.year


def spread(g, v):
    return pd.DataFrame({s: (R[v][s] - R[g][s]) * BP for s in ("on", "id", "cc")}, index=IDX)


SP = {(ix, fam): spread(*p) for ix, fams in STYLE.items() for fam, p in fams.items()}
SP_ALL = {(ix, fam): pd.DataFrame({s: (R_ALL[p[1]][s] - R_ALL[p[0]][s]) * BP for s in ("on", "id", "cc")}, index=IDX_ALL)
          for ix, fams in STYLE.items() for fam, p in fams.items()}
SPREADS = list(SP.keys())


# ------------------------------------------------------------------ helpers
def nw_var_mean(x, lags=LAGS):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return v / n


def cov_t(x, y):
    m = pd.concat([x, y], axis=1).dropna().values
    z = (m[:, 0] - m[:, 0].mean()) * (m[:, 1] - m[:, 1].mean())
    return z.mean(), z.mean() / np.sqrt(nw_var_mean(z))


def iv_slope(y, x, z):
    m = pd.concat([y, x, z], axis=1).dropna()
    yy, xx, zz = (m.iloc[:, i].values for i in range(3))
    zc, xc, yc = zz - zz.mean(), xx - xx.mean(), yy - yy.mean()
    b = (zc @ yc) / (zc @ xc)
    g = zc * (yc - b * xc)
    se = np.sqrt(nw_var_mean(g) * len(g) ** 2) / abs(zc @ xc)
    return b, b / se


def ols_hac(y, X):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})


def star(t):
    a = abs(t)
    return "$^{***}$" if a > 2.576 else "$^{**}$" if a > 1.96 else "$^{*}$" if a > 1.645 else ""


def f(x, d=2):
    if x is None or not np.isfinite(x):
        return "--"
    v = round(float(x), d) + 0.0
    if v == 0:
        v = 0.0
    return f"{v:,.{d}f}".replace("-", "$-$")


def write(name, body):
    (TAB / f"{name}.tex").write_text(body)


def lab(ix, fam):
    return f"{ix} {FAM[fam]}"


def sisters(ix, fam):
    return [k for k in STYLE[ix] if k != fam]


def sister_mean(ix, fam, s):
    return pd.concat([SP[(ix, k)][s] for k in sisters(ix, fam)], axis=1).mean(axis=1)


def table(caption, label, colspec, header, lines, wide=True, size="\\small"):
    if wide:
        return (f"\\begin{{table}}[H]\n\\caption{{{caption}\\label{{{label}}}}}\n\\begin{{adjustwidth}}{{-\\extralength}}{{0cm}}\n"
                f"\\centering\n{size}\n\\begin{{tabularx}}{{\\fulllength}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}\n")
    return (f"\\begin{{table}}[H]\n\\caption{{{caption}\\label{{{label}}}}}\n\\centering\n{size}\n"
            f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")


SIG = "Significance from Newey--West ($10$ lags) $t$-statistics: $^{*}$ 10\\%, $^{**}$ 5\\%, $^{***}$ 1\\%."
SAMPLE = "Sample: 3 January 2011 to 25 September 2026, excluding 24 August 2015"
print("same-index sample", IDX[0].date(), IDX[-1].date(), len(IDX))

# ================================================================== fund list (appendix)
rows = []
for ix in STYLE:
    for fam in STYLE[ix]:
        for t, kind in zip(STYLE[ix][fam], ("Growth", "Value")):
            rows.append((ix, fam, t, kind))
    for fam, t in BENCH[ix].items():
        rows.append((ix, fam, t, "Blend"))
recs = []
for ix, fam, t, kind in rows:
    r = R[t]
    e, l = r.loc[:"2015"], r.loc["2021":]
    recs.append(dict(index=ix, fam=fam, t=t, kind=kind, first=RAW[t]["first"].iloc[0].date(),
                     dv_e=e.dvol.mean() / 1e6, dv_l=l.dvol.mean() / 1e6, ext_e=e.ext.mean() * 100,
                     ext_l=l.ext.mean() * 100, stale_e=e.stale.mean() * 100, stale_l=l.stale.mean() * 100))
FL = pd.DataFrame(recs)
FL.to_csv(RES / "fund_list.csv", index=False)
lines = []
for ix in STYLE:
    for _, x in FL[FL["index"] == ix].iterrows():
        lines.append(f"{x.t} & {INDEXNAME[ix]} {x.kind if x.kind != 'Blend' else ''} & {FAM[x.fam]} & {x['first']} & "
                     f"{x.dv_e:,.1f} & {x.dv_l:,.1f} & {f(x.ext_e,1)} & {f(x.ext_l,1)} & {f(x.stale_e,1)} & {f(x.stale_l,1)} \\\\")
    lines.append("\\midrule")
write("tab_funds", table(
    "Funds in the same-index sample. First is the first trading day in the data. Dollar volume is the average daily closing price times volume in millions of US dollars; Extreme is the percentage of days on which the open equals the day's high or low; Stale is the percentage of days on which the open equals the previous close. Early: 2011--2015; late: 2021 to 25 September 2026.",
    "tab:funds", "lllcCCCCCC",
    " & & & & \\multicolumn{2}{c}{\\textbf{Dollar volume}} & \\multicolumn{2}{c}{\\textbf{Extreme (\\%)}} & \\multicolumn{2}{c}{\\textbf{Stale (\\%)}}\\\\\n"
    "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
    "\\textbf{ETF} & \\textbf{Index} & \\textbf{Family} & \\textbf{First} & \\textbf{Early} & \\textbf{Late} & \\textbf{Early} & \\textbf{Late} & \\textbf{Early} & \\textbf{Late}\\\\",
    lines[:-1], size="\\footnotesize"))

# family summary for text
fs = FL.groupby("fam")[["dv_e", "dv_l", "ext_e", "ext_l", "stale_e", "stale_l"]].median()
fs.to_csv(RES / "family_summary.csv")

# ================================================================== same-index sister correlations
rows = []
for ix in STYLE:
    fams = list(STYLE[ix])
    rec = dict(index=ix, nfam=len(fams))
    for per, m in [("full", np.ones(len(IDX), bool)), ("early", YEARS <= 2015), ("late", YEARS >= 2021)]:
        rn = np.mean([SP[(ix, a)].on[m].corr(SP[(ix, b)].on[m]) for a, b in combinations(fams, 2)])
        rd = np.mean([SP[(ix, a)].id[m].corr(SP[(ix, b)].id[m]) for a, b in combinations(fams, 2)])
        rc = np.mean([SP[(ix, a)].cc[m].corr(SP[(ix, b)].cc[m]) for a, b in combinations(fams, 2)])
        rec.update({f"{per}_on": rn, f"{per}_id": rd, f"{per}_cc": rc})
    rec["sd_on"] = np.mean([SP[(ix, a)].on.std() for a in fams])
    rec["sd_id"] = np.mean([SP[(ix, a)].id.std() for a in fams])
    rows.append(rec)
SC = pd.DataFrame(rows)
SC.to_csv(RES / "same_corr.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} & {x.nfam} & {f(x.sd_on,1)} & {f(x.sd_id,1)} & {f(x.full_on)} & {f(x.full_id)} & {f(x.full_cc)} & "
         f"{f(x.early_on)} & {f(x.early_id)} & {f(x.late_on)} & {f(x.late_id)} \\\\" for _, x in SC.iterrows()]
write("tab_samecorr", table(
    "Session spreads of funds tracking the same index. For each style index, the value-minus-growth spread is computed for each fund family and session. Std.\\ dev.\\ is the average daily standard deviation across families, in basis points. Correlations are averaged over all pairs of families tracking the same index; CC is the close-to-close spread. Early: 2011--2015; late: 2021 to 25 September 2026. " + SAMPLE + " (3955 days).",
    "tab:samecorr", "lcCCCCCCCCC",
    " & & \\multicolumn{2}{c}{\\textbf{Std. dev. (bps)}} & \\multicolumn{3}{c}{\\textbf{Correlation, full}} & \\multicolumn{2}{c}{\\textbf{Early}} & \\multicolumn{2}{c}{\\textbf{Late}}\\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n"
    "\\textbf{Index} & \\textbf{Families} & \\textbf{On} & \\textbf{Id} & \\textbf{On} & \\textbf{Id} & \\textbf{CC} & \\textbf{On} & \\textbf{Id} & \\textbf{On} & \\textbf{Id}\\\\",
    lines))


# ================================================================== main same-index test
def same_test(a: pd.DataFrame, dB: pd.Series, nB: pd.Series) -> dict:
    n, d = a.on, a.id
    co, to = cov_t(n, d)
    cx, tx = cov_t(n, dB)
    s2, ts2 = cov_t(n, dB - d)
    ols = ols_hac(d, n)
    biv, tiv = iv_slope(d, n, nB)
    return dict(rho_own=n.corr(d), rho_x=n.corr(dB), cov_own=co, t_own=to, cov_x=cx, t_x=tx,
                b_ols=ols.params.iloc[1], t_ols=ols.tvalues.iloc[1], b_iv=biv, t_iv=tiv,
                s2e=s2, t_s2e=ts2, share_var=s2 / n.var(), share_cov=s2 / -co if co < 0 else np.nan, var_on=n.var())


rows = []
for ix, fam in SPREADS:
    x = same_test(SP[(ix, fam)], sister_mean(ix, fam, "id"), sister_mean(ix, fam, "on"))
    x.update(index=ix, fam=fam, sis="/".join(FAM[k] for k in sisters(ix, fam)))
    rows.append(x)
ST = pd.DataFrame(rows)
ST.to_csv(RES / "same_test.csv", index=False)
lines = []
for ix in STYLE:
    for _, x in ST[ST["index"] == ix].iterrows():
        lines.append(f"{INDEXNAME[ix]} & {FAM[x.fam]} & {f(x.rho_own)} & {f(x.rho_x)} & {f(x.cov_own,0)}{star(x.t_own)} & {f(x.cov_x,0)}{star(x.t_x)} & "
                     f"{f(x.b_ols)}{star(x.t_ols)} & {f(x.b_iv)}{star(x.t_iv)} & {f(x.s2e,0)}{star(x.t_s2e)} & {f(np.sqrt(max(x.s2e, 0)),1)} & {f(100*x.share_var,0)} \\\\")
    lines.append("\\midrule")
write("tab_same", table(
    "Own-fund and same-index sister reversal. For each value-minus-growth spread $A$, own measures relate its overnight spread $n_A$ to its own intraday spread $d_A$; cross measures relate $n_A$ to $\\bar d_B$, the average intraday spread of the other fund families tracking the same pair of style indices. Cov is in squared basis points. OLS is the slope of $d_A$ on $n_A$; IV instruments $n_A$ with the average sister overnight spread $\\bar n_B$. $\\hat\\sigma^2_e = \\operatorname{Cov}(n_A, \\bar d_B - d_A)$ is the variance of the fund-specific opening error of the spread; $\\hat\\sigma_e$ is its square root in basis points, and the last column expresses $\\hat\\sigma^2_e$ as a percentage of the variance of $n_A$. " + SIG + " " + SAMPLE + ".",
    "tab:same", "llCCCCCCCCC",
    " & & \\multicolumn{2}{c}{$\\boldsymbol{\\rho}$} & \\multicolumn{2}{c}{\\textbf{Cov (bps$^2$)}} & \\multicolumn{2}{c}{\\textbf{Slope}} & & & \\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
    "\\textbf{Index} & \\textbf{Family} & \\textbf{Own} & \\textbf{Cross} & \\textbf{Own} & \\textbf{Cross} & \\textbf{OLS} & \\textbf{IV} & $\\boldsymbol{\\hat\\sigma^2_e}$ & $\\boldsymbol{\\hat\\sigma_e}$ & \\textbf{\\% of Var($n_A$)}\\\\",
    lines[:-1]))

# all 22 ordered pairs (appendix)
rows = []
for ix in STYLE:
    for a, b in [(a, b) for a in STYLE[ix] for b in STYLE[ix] if a != b]:
        x = same_test(SP[(ix, a)], SP[(ix, b)].id, SP[(ix, b)].on)
        x.update(index=ix, A=a, B=b)
        rows.append(x)
PT = pd.DataFrame(rows)
PT.to_csv(RES / "same_pairs.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} & {FAM[x.A]} & {FAM[x.B]} & {f(x.rho_own)} & {f(x.rho_x)}{star(x.t_x)} & {f(x.b_ols)}{star(x.t_ols)} & {f(x.b_iv)}{star(x.t_iv)} & {f(x.s2e,0)}{star(x.t_s2e)} & {f(np.sqrt(max(x.s2e, 0)),1)} \\\\"
         for _, x in PT.iterrows()]
write("tab_pairs", table(
    "All ordered same-index pairs. As Table~\\ref{tab:same}, but with a single sister family $B$ in place of the average of the sisters. " + SIG + " " + SAMPLE + ".",
    "tab:pairs", "lllCCCCCC",
    "\\textbf{Index} & \\textbf{A} & \\textbf{B} & \\textbf{$\\rho$ own} & \\textbf{$\\rho$ cross} & \\textbf{OLS} & \\textbf{IV} & $\\boldsymbol{\\hat\\sigma^2_e}$ & $\\boldsymbol{\\hat\\sigma_e}$\\\\",
    lines, size="\\footnotesize"))

# subperiods
SUBS = [("2011--2015", 2011, 2015), ("2016--2020", 2016, 2020), ("2021--2026", 2021, 2026)]
rows = []
for ix, fam in SPREADS:
    a, dB = SP[(ix, fam)], sister_mean(ix, fam, "id")
    rec = dict(index=ix, fam=fam)
    for nm, y0, y1 in SUBS:
        m = (YEARS >= y0) & (YEARS <= y1)
        rec[nm + "_own"], rec[nm + "_town"] = a.on[m].corr(a.id[m]), cov_t(a.on[m], a.id[m])[1]
        rec[nm + "_x"], rec[nm + "_tx"] = a.on[m].corr(dB[m]), cov_t(a.on[m], dB[m])[1]
        rec[nm + "_s2"] = cov_t(a.on[m], dB[m] - a.id[m])[0]
    rows.append(rec)
SS = pd.DataFrame(rows)
SS.to_csv(RES / "same_sub.csv", index=False)
lines = []
for _, x in SS.iterrows():
    cells = []
    for nm, _, _ in SUBS:
        cells += [f(x[nm + "_own"]) + star(x[nm + "_town"]), f(x[nm + "_x"]) + star(x[nm + "_tx"]), f(np.sqrt(max(x[nm + "_s2"], 0)), 1)]
    lines.append(f"{INDEXNAME[x['index']]} {FAM[x.fam]} & " + " & ".join(cells) + " \\\\")
write("tab_samesub", table(
    "Same-index own and cross correlations by subperiod. Own is $\\rho(n_A, d_A)$, cross is $\\rho(n_A, \\bar d_B)$, and $\\hat\\sigma_e$ is the square root of the noise variance estimate in basis points (negative estimates set to zero). " + SIG,
    "tab:samesub", "lCCCCCCCCC",
    " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{nm}}}}}" for nm, _, _ in SUBS) + "\\\\\n"
    + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(3)) + "\n"
    "\\textbf{Spread} & " + " & ".join(["\\textbf{Own} & \\textbf{Cross} & $\\boldsymbol{\\hat\\sigma_e}$"] * 3) + "\\\\",
    lines, size="\\footnotesize"))


# ================================================================== fund-level noise: three-cornered hat
def pair_open(a, b, m=None):
    """-Cov(n_a-n_b, d_a-d_b) = sigma2_e(a)+sigma2_e(b) in bps^2."""
    na, nb, da, db = (R[a].on * BP, R[b].on * BP, R[a].id * BP, R[b].id * BP)
    if m is not None:
        na, nb, da, db = na[m], nb[m], da[m], db[m]
    return -cov_t(na - nb, da - db)[0]


def pair_close(a, b, m=None):
    """-Cov(d_a-d_b at t, n_a-n_b at t+1) = sigma2_u(a)+sigma2_u(b)."""
    dd = (R[a].id - R[b].id) * BP
    nn = ((R[a].on - R[b].on) * BP).shift(-1)
    if m is not None:
        dd, nn = dd[m], nn[m]
    return -cov_t(dd, nn)[0]


def hat(funds, fn, m=None):
    a, b, c = funds
    sab, sac, sbc = fn(a, b, m), fn(a, c, m), fn(b, c, m)
    return {a: (sab + sac - sbc) / 2, b: (sab + sbc - sac) / 2, c: (sac + sbc - sab) / 2}


GROUPS = []  # (label, {fam: ticker})
for ix in STYLE:
    for j, kind in enumerate(("Growth", "Value")):
        GROUPS.append((f"{INDEXNAME[ix]} {kind}", ix, {fam: STYLE[ix][fam][j] for fam in STYLE[ix]}))
    GROUPS.append((f"{INDEXNAME[ix]}", ix, dict(BENCH[ix])))
rows = []
for gl, ix, fams in GROUPS:
    tick = list(fams.values())
    rec = dict(group=gl, index=ix, n=len(tick))
    if len(tick) == 3:
        o = hat(tick, pair_open)
        c = hat(tick, pair_close)
        oe = hat(tick, pair_open, YEARS <= 2015)
        ol = hat(tick, pair_open, YEARS >= 2021)
        for fam, t in fams.items():
            rec.update({f"{fam}_t": t, f"{fam}_o": o[t], f"{fam}_c": c[t], f"{fam}_oe": oe[t], f"{fam}_ol": ol[t]})
    else:
        t1, t2 = tick
        rec.update(pair_o=pair_open(t1, t2), pair_c=pair_close(t1, t2),
                   pair_oe=pair_open(t1, t2, YEARS <= 2015), pair_ol=pair_open(t1, t2, YEARS >= 2021))
        for fam, t in fams.items():
            rec[f"{fam}_t"] = t
    rows.append(rec)
FN = pd.DataFrame(rows)
FN.to_csv(RES / "fund_noise.csv", index=False)


def sd(v):
    return np.sign(v) * np.sqrt(abs(v)) if np.isfinite(v) else np.nan


def sdf(v):
    if not np.isfinite(v):
        return "--"
    return f(sd(v), 1) if v >= 0 else "$<$0"


lines = []
for _, x in FN.iterrows():
    if x.n == 3:
        cells = []
        for col in ("o", "c", "oe", "ol"):
            cells += [sdf(x[f"{fam}_{col}"]) for fam in ("I", "V", "S")]
        lines.append(f"{x.group} & " + " & ".join(cells) + " \\\\")
    else:
        cells = []
        for col in ("pair_o", "pair_c", "pair_oe", "pair_ol"):
            cells += [f"\\multicolumn{{3}}{{c}}{{{sdf(x[col])} (sum of I and V)}}"]
        lines.append(f"{x.group} & " + " & ".join(cells) + " \\\\")
write("tab_fundnoise", table(
    "Fund-level opening and closing price noise. Standard deviations, in basis points, of fund-specific errors in the opening price (Open) and the closing price (Close) of individual ETFs, for iShares (I), Vanguard (V) and SPDR (S) funds tracking the same index. For groups of three funds, each fund's error variance is identified from the three pairwise sums $\\sigma^2_A + \\sigma^2_B = -\\operatorname{Cov}(n_A - n_B, d_A - d_B)$ for the open and $-\\operatorname{Cov}(d_{A,t} - d_{B,t}, n_{A,t+1} - n_{B,t+1})$ for the close (Equation~(\\ref{eq:hat})); for groups of two funds only the sum is identified. $<$0: negative variance estimate. Full: 2011--2026; early: 2011--2015; late: 2021 to 25 September 2026; 24 August 2015 excluded.",
    "tab:fundnoise", "lCCCCCCCCCCCC",
    " & \\multicolumn{3}{c}{\\textbf{Open, full}} & \\multicolumn{3}{c}{\\textbf{Close, full}} & \\multicolumn{3}{c}{\\textbf{Open, early}} & \\multicolumn{3}{c}{\\textbf{Open, late}}\\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n"
    "\\textbf{Index fund group} & " + " & ".join(["\\textbf{I} & \\textbf{V} & \\textbf{S}"] * 4) + "\\\\",
    lines, size="\\footnotesize"))

# ================================================================== fund-year panel: noise vs liquidity
rows = []
for gl, ix, fams in GROUPS:
    tick = list(fams.values())
    if len(tick) != 3:
        continue
    for y in range(2011, 2027):
        m = YEARS == y
        o = hat(tick, pair_open, m)
        for fam, t in fams.items():
            rows.append(dict(group=gl, index=ix, year=y, fam=fam, t=t, s2=o[t],
                             ldv=np.log(R[t].dvol[m].mean()), ext=R[t].ext[m].mean() * 100,
                             stale=R[t].stale[m].mean() * 100))
FP = pd.DataFrame(rows)
FP["sd"] = np.sign(FP.s2) * np.sqrt(FP.s2.abs())
FP["gy"] = FP.group + "_" + FP.year.astype(str)
FP.to_csv(RES / "fund_panel.csv", index=False)
res = []
for ycol, ylab in [("s2", "Noise variance (bps$^2$)"), ("sd", "Signed noise s.d. (bps)")]:
    for spec, xcols in [("Dollar volume", ["ldv"]), ("Extreme opens", ["ext"]), ("Both", ["ldv", "ext"])]:
        for fe in ("gy", "g+y"):
            X = FP[xcols].copy()
            if fe == "gy":
                X = X.join(pd.get_dummies(FP.gy, prefix="gy", drop_first=True, dtype=float))
            else:
                X = X.join(pd.get_dummies(FP.group, prefix="g", drop_first=True, dtype=float)).join(
                    pd.get_dummies(FP.year, prefix="y", drop_first=True, dtype=float))
            m = sm.OLS(FP[ycol], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": FP.year})
            rec = dict(y=ylab, spec=spec, fe=fe, r2=m.rsquared, n=int(m.nobs))
            for c in ("ldv", "ext"):
                rec[c], rec["t_" + c] = (m.params.get(c, np.nan), m.tvalues.get(c, np.nan))
            res.append(rec)
FR = pd.DataFrame(res)
FR.to_csv(RES / "fund_liq.csv", index=False)
lines = []
for ylab in FR.y.unique():
    first = True
    for _, x in FR[(FR.y == ylab) & (FR.fe == "gy")].iterrows():
        lines.append(f"{ylab if first else ''} & {x.spec} & {f(x.ldv,2) + star(x.t_ldv) if np.isfinite(x.ldv) else ''} & "
                     f"{'(' + f(x.t_ldv) + ')' if np.isfinite(x.ldv) else ''} & {f(x.ext,2) + star(x.t_ext) if np.isfinite(x.ext) else ''} & "
                     f"{'(' + f(x.t_ext) + ')' if np.isfinite(x.ext) else ''} & {f(x.r2)} \\\\")
        first = False
    lines.append("\\midrule")
write("tab_fundliq", table(
    "Fund-level opening noise and liquidity within index and year. Each observation is one fund in one calendar year, for the eight groups of three funds (iShares, Vanguard, SPDR) tracking the same index, 2011--2026 ($n = 384$; 24 August 2015 excluded). The dependent variable is the fund's opening-noise variance from the three-cornered-hat decomposition, or its signed square root. Regressors are the log of the fund's average daily dollar volume and its percentage of days with the open at the day's high or low. All regressions include group-by-year fixed effects, so the coefficients compare funds tracking the same index in the same year. $t$-statistics clustered by year in parentheses; $^{*}$ 10\\%, $^{**}$ 5\\%, $^{***}$ 1\\%.",
    "tab:fundliq", "llCCCCC",
    "\\textbf{Dependent} & \\textbf{Specification} & \\textbf{Log dollar vol.} & $\\boldsymbol{t}$ & \\textbf{Extreme opens (\\%)} & $\\boldsymbol{t}$ & $\\boldsymbol{R^2}$\\\\",
    lines[:-1], wide=False))

# ================================================================== print quality (same-index pooled)
panel = []
for k, (ix, fam) in enumerate(SPREADS):
    g, v = STYLE[ix][fam]
    a, dB = SP[(ix, fam)], sister_mean(ix, fam, "id")
    q = pd.DataFrame(index=IDX)
    q["stale"] = ((R[g].stale + R[v].stale) > 0).astype(float)
    q["ext"] = ((R[g].ext + R[v].ext) > 0).astype(float)
    q["lowvol"] = (pd.concat([R[g].relvol, R[v].relvol], axis=1).min(axis=1) < 0.5).astype(float)
    df = pd.DataFrame({"pair": k, "n": a.on, "d": a.id, "dS": dB}, index=IDX).join(q)
    df["date"] = np.arange(len(df))
    panel.append(df)
P = pd.concat(panel).dropna().reset_index(drop=True)
rows = []
for flag, nm in [("ext", "Open at high/low"), ("stale", "Stale open"), ("lowvol", "Low volume")]:
    rec = dict(flag=nm, freq=P[flag].mean() * 100)
    for y, tag in [("d", "own"), ("dS", "cross")]:
        X = pd.DataFrame({"n": P.n, "nQ": P.n * P[flag], "Q": P[flag]}).join(pd.get_dummies(P.pair, prefix="p", drop_first=True, dtype=float))
        m = sm.OLS(P[y], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": P.date})
        rec.update({f"{tag}_b": m.params["n"], f"{tag}_tb": m.tvalues["n"], f"{tag}_c": m.params["nQ"], f"{tag}_tc": m.tvalues["nQ"]})
    rows.append(rec)
SQ = pd.DataFrame(rows)
SQ.to_csv(RES / "same_quality.csv", index=False)
lines = []
for _, x in SQ.iterrows():
    lines.append(f"{x.flag} & {f(x.freq,1)} & {f(x.own_b,3)}{star(x.own_tb)} & {f(x.own_c,3)}{star(x.own_tc)} & {f(x.cross_b,3)}{star(x.cross_tb)} & {f(x.cross_c,3)}{star(x.cross_tc)} \\\\")
    lines.append(f" & & ({f(x.own_tb)}) & ({f(x.own_tc)}) & ({f(x.cross_tb)}) & ({f(x.cross_tc)}) \\\\")
write("tab_samequal", table(
    "Reversal conditional on opening-print quality, same-index sample. Pooled regressions over the 13 spreads of $y_t = a_k + b\\,n_{A,t} + c\\,n_{A,t} Q_{A,t} + g\\,Q_{A,t} + u_t$ with spread fixed effects $a_k$. Own: $y = d_A$; cross: $y = \\bar d_B$, the average intraday spread of the same-index sister families. Flags as in Table~\\ref{tab:quality}. $t$-statistics clustered by date in parentheses; $^{*}$ 10\\%, $^{**}$ 5\\%, $^{***}$ 1\\%. " + SAMPLE + ".",
    "tab:samequal", "lCCCCC",
    " & & \\multicolumn{2}{c}{\\textbf{Own ($y=d_A$)}} & \\multicolumn{2}{c}{\\textbf{Cross ($y=\\bar d_B$)}}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
    "\\textbf{Flag $Q$} & \\textbf{Freq. (\\%)} & $\\boldsymbol{b}$ & $\\boldsymbol{c}$ & $\\boldsymbol{b}$ & $\\boldsymbol{c}$\\\\",
    lines, wide=False))

# ================================================================== session risk (same-index)
rows = []
for ix, fam in SPREADS:
    a, dB = SP[(ix, fam)], sister_mean(ix, fam, "id")
    s2 = cov_t(a.on, dB - a.id)[0]
    vn, vd, c, vcc = a.on.var(), a.id.var(), a.on.cov(a.id), a.cc.var()
    rows.append(dict(index=ix, fam=fam, s2e=s2, vol_cc=np.sqrt(vcc * 252) / 100,
                     vol_on=np.sqrt(vn * 252) / 100, vol_on_c=np.sqrt(max(vn - s2, 0) * 252) / 100,
                     vol_id=np.sqrt(vd * 252) / 100, vol_id_c=np.sqrt(max(vd - s2, 0) * 252) / 100,
                     sh=vn / vcc, sh_c=(vn - s2) / vcc, rho=c / np.sqrt(vn * vd),
                     rho_c=(c + s2) / np.sqrt((vn - s2) * (vd - s2))))
SR = pd.DataFrame(rows)
SR.to_csv(RES / "same_risk.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} {FAM[x.fam]} & {f(x.vol_cc)} & {f(x.vol_on)} & {f(x.vol_on_c)} & {f(x.vol_id)} & {f(x.vol_id_c)} & {f(100*x.sh,0)} & {f(100*x.sh_c,0)} & {f(x.rho)} & {f(x.rho_c)} \\\\"
         for _, x in SR.iterrows()]
write("tab_samerisk", table(
    "Session risk of style spreads before and after removing opening-price noise, same-index sample. As Table~\\ref{tab:risk}, with the noise variance estimated from same-index sister funds (Table~\\ref{tab:same}). Volatilities are annualized, in percent. " + SAMPLE + ".",
    "tab:samerisk", "lCCCCCCCCC",
    " & & \\multicolumn{2}{c}{\\textbf{Overnight vol.}} & \\multicolumn{2}{c}{\\textbf{Intraday vol.}} & \\multicolumn{2}{c}{\\textbf{On share (\\%)}} & \\multicolumn{2}{c}{$\\boldsymbol{\\rho}$\\textbf{(on, id)}}\\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
    "\\textbf{Spread} & \\textbf{CC vol.} & \\textbf{Raw} & \\textbf{Corr.} & \\textbf{Raw} & \\textbf{Corr.} & \\textbf{Raw} & \\textbf{Corr.} & \\textbf{Raw} & \\textbf{Corr.}\\\\",
    lines))


# ================================================================== variance forecasts with a robust proxy
def rolling_cov(x, y, w=WIN):
    return (x * y).rolling(w).mean().shift(1) - (x.rolling(w).mean() * y.rolling(w).mean()).shift(1)


FC = {}
rows = []
for ix, fam in SPREADS:
    a = SP[(ix, fam)]
    nA, dA = a.on, a.id
    nB, dB = sister_mean(ix, fam, "on"), sister_mean(ix, fam, "id")
    raw = nA.rolling(WIN).var().shift(1)
    s2r = rolling_cov(nA, dB - dA)                       # opening-noise variance
    # closing-noise variance: Cov(d_A,t, nbar_B,t+1 - n_A,t+1), dated so only past data are used
    s2c = rolling_cov(dA.shift(1), (nB - nA))
    cor = (raw - s2r).clip(lower=0.05 * raw)
    cor2 = (raw - s2r - s2c).clip(lower=0.05 * raw)
    proxy = nA * nB  # unbiased for Var(n*) if errors are independent across funds
    df = pd.DataFrame({"raw": raw, "cor": cor, "cor2": cor2, "proxy": proxy}).dropna()
    FC[(ix, fam)] = df
    lr, lc, lc2 = (df.proxy - df.raw) ** 2, (df.proxy - df.cor) ** 2, (df.proxy - df.cor2) ** 2
    dm, dm2 = (lr - lc), (lr - lc2)
    rows.append(dict(index=ix, fam=fam, n=len(df), proxy=df.proxy.mean(), raw=df.raw.mean(), cor=df.cor.mean(),
                     cor2=df.cor2.mean(),
                     bias_raw=df.raw.mean() / df.proxy.mean() - 1, bias_cor=df.cor.mean() / df.proxy.mean() - 1,
                     bias_cor2=df.cor2.mean() / df.proxy.mean() - 1,
                     mse_ratio=lc.mean() / lr.mean(), dm_t=dm.mean() / np.sqrt(nw_var_mean(dm.values)),
                     mse_ratio2=lc2.mean() / lr.mean(), dm_t2=dm2.mean() / np.sqrt(nw_var_mean(dm2.values))))
FT = pd.DataFrame(rows)
FT.to_csv(RES / "forecast.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} {FAM[x.fam]} & {f(x.proxy,0)} & {f(x.raw,0)} & {f(x.cor,0)} & {f(x.cor2,0)} & {f(100*x.bias_raw,0)} & {f(100*x.bias_cor,0)} & {f(100*x.bias_cor2,0)} & {f(x.mse_ratio,2)} & {f(x.dm_t)}{star(x.dm_t)} & {f(x.mse_ratio2,2)} & {f(x.dm_t2)}{star(x.dm_t2)} \\\\"
         for _, x in FT.iterrows()]
write("tab_forecast", table(
    "Forecasting the overnight variance of style spreads. Each day, the variance of the next overnight spread is forecast from the previous 250 days, either from reported prices (Raw), after subtracting the rolling opening-noise variance $\\widehat{\\operatorname{Cov}}(n_A, \\bar d_B - d_A)$ (Open), or after also subtracting the rolling closing-noise variance $\\widehat{\\operatorname{Cov}}(d_{A,t}, \\bar n_{B,t+1} - n_{A,t+1})$ (Full). Forecasts are evaluated against the proxy $n_{A,t}\\bar n_{B,t}$, which is unbiased for the efficient overnight variance when pricing errors are independent across funds; squared-error loss is robust to the noise in such a proxy \\citep{Patton2011}. Means are in squared basis points; bias is the mean forecast relative to the mean proxy, in percent. MSE ratio is the mean squared error of a corrected forecast divided by that of the raw forecast; DM is the Diebold--Mariano $t$-statistic for equal loss (positive: corrected better). " + SIG + " " + SAMPLE + "; evaluation from January 2012.",
    "tab:forecast", "lCCCCCCCCCCC",
    " & \\multicolumn{4}{c}{\\textbf{Mean (bps$^2$)}} & \\multicolumn{3}{c}{\\textbf{Bias (\\%)}} & \\multicolumn{2}{c}{\\textbf{Open vs.\\ raw}} & \\multicolumn{2}{c}{\\textbf{Full vs.\\ raw}}\\\\\n"
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\n"
    "\\textbf{Spread} & \\textbf{Proxy} & \\textbf{Raw} & \\textbf{Open} & \\textbf{Full} & \\textbf{Raw} & \\textbf{Open} & \\textbf{Full} & \\textbf{MSE ratio} & \\textbf{DM} & \\textbf{MSE ratio} & \\textbf{DM}\\\\",
    lines, size="\\footnotesize"))


# ================================================================== VaR backtests
def kupiec(x: int, n: int, p: float) -> float:
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return 1 - stats.chi2.cdf(lr, 1)


rows = []
z = stats.norm.ppf(0.99)
for ix, fam in SPREADS:
    df = FC[(ix, fam)]
    nA = SP[(ix, fam)].on.reindex(df.index)
    rec = dict(index=ix, fam=fam, n=len(df))
    vr, vc = z * np.sqrt(df.raw), z * np.sqrt(df.cor)
    rec["var_raw"], rec["var_cor"] = vr.mean(), vc.mean()
    rec["over"] = (vr / vc - 1).mean()
    # efficient-variance VaR implied by the proxy (250-day mean of n_A * nbar_B)
    for tag, v in (("raw", vr), ("cor", vc)):
        x = int(((-nA > v) | (nA > v)).sum())  # long and short spread positions
        rec[f"{tag}_exc"] = x / len(df) / 2 * 100
        rec[f"{tag}_p"] = kupiec(x, 2 * len(df), 0.01)
    rows.append(rec)
VT = pd.DataFrame(rows)
VT.to_csv(RES / "var.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} {FAM[x.fam]} & {f(x.var_raw,0)} & {f(x.var_cor,0)} & {f(100*x.over,0)} & {f(x.raw_exc)} & {f(x.cor_exc)} \\\\" for _, x in VT.iterrows()]
avg = VT[["var_raw", "var_cor", "over", "raw_exc", "cor_exc"]].mean()
lines.append("\\midrule")
lines.append(f"Average & {f(avg.var_raw,0)} & {f(avg.var_cor,0)} & {f(100*avg.over,0)} & {f(avg.raw_exc)} & {f(avg.cor_exc)} \\\\")
write("tab_var", table(
    "Overnight value-at-risk of style-spread positions. Each day, a Gaussian 99\\% VaR for the next overnight spread is computed from the raw or the noise-corrected 250-day variance forecast of Table~\\ref{tab:forecast}. VaR columns report averages in basis points of the notional of one leg; Overstatement is the average percentage by which the raw VaR exceeds the corrected VaR. Exceedances are the percentage of position-days, for long and short spread positions, on which the fund's own reported overnight spread $n_A$ falls beyond the VaR (nominal: 1\\%). The reported spread is the outcome realized by a position traded at the opening price, for which opening noise is a genuine risk. " + SAMPLE + "; evaluation from January 2012.",
    "tab:var", "lCCCCC",
    " & \\multicolumn{2}{c}{\\textbf{99\\% VaR (bps)}} & & \\multicolumn{2}{c}{\\textbf{Exceedances vs.\\ $n_A$ (\\%)}}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\n"
    "\\textbf{Spread} & \\textbf{Raw} & \\textbf{Corrected} & \\textbf{Overstatement (\\%)} & \\textbf{Raw VaR} & \\textbf{Corr. VaR}\\\\",
    lines, wide=False))


# ================================================================== paper profits, same index
def ls(sig, ret):
    p = (-np.sign(sig) * ret).dropna()
    return p.mean() * 252 / 100, p.mean() / p.std() * np.sqrt(252), p.mean() / np.sqrt(nw_var_mean(p.values))


rows = []
for ix, fam in SPREADS:
    a, nB = SP[(ix, fam)], sister_mean(ix, fam, "on")
    rec = dict(index=ix, fam=fam)
    for per, m in [("full", np.ones(len(IDX), bool)), ("late", YEARS >= 2021)]:
        o, x = ls(a.on[m], a.id[m]), ls(nB[m], a.id[m])
        rec.update({f"{per}_own_mu": o[0], f"{per}_own_sr": o[1], f"{per}_own_t": o[2],
                    f"{per}_x_mu": x[0], f"{per}_x_sr": x[1], f"{per}_x_t": x[2]})
    rows.append(rec)
PP = pd.DataFrame(rows)
PP.to_csv(RES / "same_profits.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} {FAM[x.fam]} & {f(x.full_own_mu,1)} & {f(x.full_own_sr)}{star(x.full_own_t)} & {f(x.full_x_mu,1)} & {f(x.full_x_sr)}{star(x.full_x_t)} & "
         f"{f(x.late_own_mu,1)} & {f(x.late_own_sr)}{star(x.late_own_t)} & {f(x.late_x_mu,1)} & {f(x.late_x_sr)}{star(x.late_x_t)} \\\\" for _, x in PP.iterrows()]
write("tab_sameprof", table(
    "Gross intraday reversal returns with own and same-index sister signals. As Table~\\ref{tab:profits}, with the sister signal $\\bar n_B$ equal to the average overnight spread of the other fund families tracking the same pair of style indices. Both signals use opening prices and are diagnostic rather than executable. Significance of the mean from Newey--West $t$-statistics: $^{*}$ 10\\%, $^{**}$ 5\\%, $^{***}$ 1\\%.",
    "tab:sameprof", "lCCCCCCCC",
    " & \\multicolumn{4}{c}{\\textbf{2011--2026}} & \\multicolumn{4}{c}{\\textbf{2021--2026}}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    " & \\multicolumn{2}{c}{\\textbf{Own signal}} & \\multicolumn{2}{c}{\\textbf{Sister signal}} & \\multicolumn{2}{c}{\\textbf{Own signal}} & \\multicolumn{2}{c}{\\textbf{Sister signal}}\\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
    "\\textbf{Spread} & \\textbf{Mean} & \\textbf{SR} & \\textbf{Mean} & \\textbf{SR} & \\textbf{Mean} & \\textbf{SR} & \\textbf{Mean} & \\textbf{SR}\\\\",
    lines))

# ================================================================== robustness (same-index)
rows = []
for ix, fam in SPREADS:
    a, dB = SP[(ix, fam)], sister_mean(ix, fam, "id")
    aa = SP_ALL[(ix, fam)]
    dBa = pd.concat([SP_ALL[(ix, k)].id for k in sisters(ix, fam)], axis=1).mean(axis=1)
    w = lambda x: x.clip(x.quantile(0.005), x.quantile(0.995))
    g, v = STYLE[ix][fam]
    nost = (R[g].stale + R[v].stale) == 0
    for k in sisters(ix, fam):
        gg, vv = STYLE[ix][k]
        nost &= (R[gg].stale + R[vv].stale) == 0
    rows.append(dict(index=ix, fam=fam,
                     ex_own=aa.on.corr(aa.id), ex_x=aa.on.corr(dBa),
                     ns_own=a.on[nost].corr(a.id[nost]), ns_x=a.on[nost].corr(dB[nost]),
                     w_own=w(a.on).corr(w(a.id)), w_x=w(a.on).corr(w(dB)),
                     sp_own=a.on.corr(a.id, method="spearman"), sp_x=a.on.corr(dB, method="spearman")))
RB = pd.DataFrame(rows)
RB.to_csv(RES / "same_robust.csv", index=False)
lines = [f"{INDEXNAME[x['index']]} {FAM[x.fam]} & " + " & ".join(f"{f(x[c+'_own'])} & {f(x[c+'_x'])}" for c in ("ex", "ns", "w", "sp")) + " \\\\" for _, x in RB.iterrows()]
write("tab_samerobust", table(
    "Robustness of the same-index own--cross contrast. Each pair of columns reports $\\rho(n_A, d_A)$ (own) and $\\rho(n_A, \\bar d_B)$ (cross) when including 24 August 2015; excluding days on which any fund of the index group has a stale open; with returns winsorized at the 0.5th and 99.5th percentiles; and with Spearman rank correlations. Sample: 3 January 2011 to 25 September 2026.",
    "tab:samerobust", "lCCCCCCCC",
    " & \\multicolumn{2}{c}{\\textbf{With 24 Aug 2015}} & \\multicolumn{2}{c}{\\textbf{No stale}} & \\multicolumn{2}{c}{\\textbf{Winsorized}} & \\multicolumn{2}{c}{\\textbf{Spearman}}\\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
    "\\textbf{Spread} & " + " & ".join(["\\textbf{Own} & \\textbf{Cross}"] * 4) + "\\\\",
    lines))

# ================================================================== figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"I": "#1f4e79", "V": "#c55a11", "S": "#5b8c3a"}

# fund-level annual opening noise, averaged over growth/value and blend funds per index
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, ix in zip(axs, ["S\\&P 500", "S\\&P 400", "S\\&P 600"]):
    sub = FP[FP["index"] == ix]
    for fam in ("I", "V", "S"):
        s = sub[sub.fam == fam].groupby("year").s2.mean()
        s = np.sign(s) * np.sqrt(s.abs())
        if len(s):
            ax.plot(s.index, s.values, "-o", ms=3, lw=1, color=COL[fam], label=FAM[fam])
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(INDEXNAME[ix].replace("\\", "") + " funds")
    ax.set_xticks(range(2012, 2027, 2))
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("Opening-price noise s.d. (bps)")
fig.tight_layout()
fig.savefig(FIG / "fig_fundnoise.pdf")
fig.savefig(FIG / "fig_fundnoise.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(11, 6.0), sharey=True)
axs = axs.ravel()
for ax, ix in zip(axs, STYLE):
    for fam in STYLE[ix]:
        a, dB = SP[(ix, fam)], sister_mean(ix, fam, "id")
        ax.plot(IDX, a.on.rolling(252).corr(a.id), "-", lw=0.9, color=COL[fam], label=f"Own, {FAM[fam]}")
        ax.plot(IDX, a.on.rolling(252).corr(dB), ":", lw=1.1, color=COL[fam], label=f"Cross, {FAM[fam]}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(INDEXNAME[ix].replace("\\", ""), fontsize=9)
axs[0].set_ylabel("Rolling 252-day correlation")
axs[3].set_ylabel("Rolling 252-day correlation")
h, l = axs[2].get_legend_handles_labels()
axs[5].axis("off")
axs[5].legend(h, l, frameon=False, fontsize=9, loc="center")
fig.tight_layout()
fig.savefig(FIG / "fig_samerolling.pdf")
fig.savefig(FIG / "fig_samerolling.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, key in zip(axs, [("S\\&P 400", "V"), ("S\\&P 600", "S")]):
    df = FC[key]
    pr = df.proxy.rolling(250).mean()
    ax.plot(df.index, np.sqrt(df.raw * 252) / 100, color="#555555", lw=1, label="Raw forecast")
    ax.plot(df.index, np.sqrt(df.cor * 252) / 100, color=COL["V"], lw=1, label="Open-noise corrected")
    ax.plot(df.index, np.sqrt(df.cor2 * 252) / 100, color=COL["S"], lw=1, label="Open- and close-noise corrected")
    ax.plot(pr.index, np.sqrt(pr.clip(lower=0) * 252) / 100, color=COL["I"], lw=1, ls="--", label="Realized proxy (250-day mean)")
    ax.set_title(f"{INDEXNAME[key[0]]} {FAM[key[1]]} V$-$G spread".replace("\\", ""))
    ax.set_ylabel("Annualized overnight volatility (%)")
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_forecast.pdf")
fig.savefig(FIG / "fig_forecast.png", dpi=150)
plt.close(fig)
print("done")
