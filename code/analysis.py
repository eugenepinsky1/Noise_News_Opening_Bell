"""Opening-print noise and the session reversal in growth-value ETF spreads.
Part 1: long-sample (2001-2026) evidence from Russell vs S&P sister spreads.
Part 2 (same-index funds, fund-level noise, risk tests) is in same_index.py.

Reads daily Yahoo Finance files in ../data (iShares, with Open/High/Low/Close,
Adj Close, Volume) and writes
CSV results to ../results, LaTeX tables to ../tables and figures to ../figures.

Session returns (as in the companion literature)
  cc = AdjClose_t / AdjClose_{t-1} - 1      close-to-close total return
  id = Close_t / Open_t - 1                 intraday (open-to-close)
  on = (1 + cc) / (1 + id) - 1              overnight (close-to-open, incl. dividends)
Spread  s = r(value) - r(growth), computed session by session.

Identification.  Each Russell style spread (A) has an S&P "sister" spread (B)
covering the same size segment.  A fund-specific error e_A in A's opening
price enters on_A with + sign and id_A with - sign, so
    cov(on_A, id_A) = c* - var(e_A),   cov(on_A, id_B) = c*_AB,
and var(e_A) is estimated by  -cov(on_A, id_A - id_B).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, TAB, FIG = ROOT / "data", ROOT / "results", ROOT / "tables", ROOT / "figures"
for p in (RES, TAB, FIG):
    p.mkdir(exist_ok=True)

START, END = "2001-09-04", "2026-09-25"
LAGS = 10
BP = 1e4
SUBS = [("2001--2007", 2001, 2007), ("2008--2014", 2008, 2014),
        ("2015--2019", 2015, 2019), ("2020--2026", 2020, 2026)]

# segment: (Russell growth, value, benchmark), (S&P growth, value, benchmark)
SEG = {
    "Large": (("IWF", "IWD", "IWB"), ("IVW", "IVE", "IVV")),
    "Mid": (("IWP", "IWS", "IWR"), ("IJK", "IJJ", "IJH")),
    "Small": (("IWO", "IWN", "IWM"), ("IJT", "IJS", "IJR")),
}
NAME = {"Large": ("Russell 1000", "S\\&P 500"), "Mid": ("Russell Midcap", "S\\&P 400"),
        "Small": ("Russell 2000", "S\\&P 600")}
INDEX = {
    "IWB": "Russell 1000", "IWF": "Russell 1000 Growth", "IWD": "Russell 1000 Value",
    "IWR": "Russell Midcap", "IWP": "Russell Midcap Growth", "IWS": "Russell Midcap Value",
    "IWM": "Russell 2000", "IWO": "Russell 2000 Growth", "IWN": "Russell 2000 Value",
    "IVV": "S\\&P 500", "IVW": "S\\&P 500 Growth", "IVE": "S\\&P 500 Value",
    "IJH": "S\\&P MidCap 400", "IJK": "S\\&P MidCap 400 Growth", "IJJ": "S\\&P MidCap 400 Value",
    "IJR": "S\\&P SmallCap 600", "IJT": "S\\&P SmallCap 600 Growth", "IJS": "S\\&P SmallCap 600 Value",
}
SISTER = {}
for seg, (ru, sp) in SEG.items():
    for a, b in zip(ru, sp):
        SISTER[a], SISTER[b] = b, a


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
    # Abdi-Ranaldo (2017) inputs and Corwin-Schultz (2012) two-day estimator
    lh, ll, lc = np.log(d.High), np.log(d.Low), np.log(d.Close)
    eta = (lh + ll) / 2
    r["ar"] = (lc - eta) * (lc - eta.shift(-1))
    r["cs"] = corwin_schultz(d)
    r["O"], r["H"], r["L"], r["C"] = d.Open, d.High, d.Low, d.Close
    return r


def corwin_schultz(d: pd.DataFrame) -> pd.Series:
    h, l, c = d.High.copy(), d.Low.copy(), d.Close
    # adjust for overnight gaps as in Corwin and Schultz (2012), Section II.B
    pc = c.shift()
    up = l > pc
    dn = h < pc
    h2 = h.where(~up, h - (l - pc)); l2 = l.where(~up, pc)
    h2 = h2.where(~dn, pc); l2 = l2.where(~dn, l + (pc - h))
    beta = np.log(h2 / l2) ** 2 + np.log(h2.shift() / l2.shift()) ** 2
    gamma = np.log(np.maximum(h2, h2.shift()) / np.minimum(l2, l2.shift())) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


TICK = sorted({t for v in SEG.values() for f in v for t in f})
R = {t: load(t) for t in TICK}
IDX = R[TICK[0]].index
for t in TICK:
    IDX = IDX.intersection(R[t].index)
IDX = IDX[(IDX >= START) & (IDX <= END)]
for t in TICK:
    R[t] = R[t].reindex(IDX)


def spread(g: str, v: str) -> pd.DataFrame:
    return pd.DataFrame({s: (R[v][s] - R[g][s]) * BP for s in ("on", "id", "cc")}, index=IDX)


SP = {}
for seg, (ru, sp) in SEG.items():
    SP[(seg, "R")] = spread(ru[0], ru[1])
    SP[(seg, "S")] = spread(sp[0], sp[1])


# ------------------------------------------------------------------ stats helpers
def nw_var_mean(x: np.ndarray, lags: int = LAGS) -> float:
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (x[k:] @ x[:-k]) / n
    return v / n


def cov_t(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Covariance and HAC t-statistic (product-series mean)."""
    m = pd.concat([x, y], axis=1).dropna().values
    z = (m[:, 0] - m[:, 0].mean()) * (m[:, 1] - m[:, 1].mean())
    return z.mean(), z.mean() / np.sqrt(nw_var_mean(z))


def ols_hac(y, X):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})


def iv_slope(y: pd.Series, x: pd.Series, z: pd.Series) -> tuple[float, float]:
    """Just-identified IV slope of y on x with instrument z, HAC s.e."""
    m = pd.concat([y, x, z], axis=1).dropna()
    yy, xx, zz = (m.iloc[:, i].values for i in range(3))
    zc, xc, yc = zz - zz.mean(), xx - xx.mean(), yy - yy.mean()
    b = (zc @ yc) / (zc @ xc)
    u = yc - b * xc
    g = zc * u
    se = np.sqrt(nw_var_mean(g) * len(g) ** 2) / abs(zc @ xc)
    return b, b / se


def star(t: float) -> str:
    a = abs(t)
    return "$^{***}$" if a > 2.576 else "$^{**}$" if a > 1.96 else "$^{*}$" if a > 1.645 else ""


def f(x, d=2):
    if not np.isfinite(x):
        return "--"
    v = round(float(x), d) + 0.0
    if v == 0:
        v = 0.0
    return f"{v:,.{d}f}".replace("-", "$-$")


def write(name: str, body: str) -> None:
    (TAB / f"{name}.tex").write_text(body)


PAIRS = []  # (seg, own fam, sister fam)
for seg in SEG:
    PAIRS.append((seg, "R", "S"))
    PAIRS.append((seg, "S", "R"))


def label(seg, fam):
    return NAME[seg][0 if fam == "R" else 1]


# ================================================================== Table 1: data
rows = []
for seg, (ru, sp) in SEG.items():
    for t in ru + sp:
        r = R[t]
        early = r.loc[:"2007"]
        late = r.loc["2020":]
        rows.append(dict(seg=seg, t=t, index=INDEX[t],
                         dvol_e=early.dvol.mean() / 1e6, dvol_l=late.dvol.mean() / 1e6,
                         stale_e=early.stale.mean() * 100, stale_l=late.stale.mean() * 100,
                         ext_e=early.ext.mean() * 100, ext_l=late.ext.mean() * 100,
                         cs_e=early.cs.mean() * BP, cs_l=late.cs.mean() * BP))
T1 = pd.DataFrame(rows)
T1.to_csv(RES / "data_summary.csv", index=False)
lines = []
for seg in SEG:
    for _, x in T1[T1.seg == seg].iterrows():
        lines.append(f"{x.t} & {x['index']} & {x.dvol_e:,.0f} & {x.dvol_l:,.0f} & {f(x.stale_e,1)} & {f(x.stale_l,1)} & "
                     f"{f(x.ext_e,1)} & {f(x.ext_l,1)} & {f(x.cs_e,1)} & {f(x.cs_l,1)} \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_data", r"""\begin{table}[H]
\caption{ETFs and indicators of opening-print quality. Dollar volume is the average daily closing price times volume in millions of US dollars. Stale is the percentage of days on which the reported open equals the previous close; Extreme is the percentage of days on which the open equals the day's high or low; CS is the Corwin--Schultz bid--ask spread estimate in basis points. Early: 4 September 2001 to 31 December 2007; late: 2 January 2020 to 25 September 2026.\label{tab:data}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{llCCCCCCCC}
\toprule
 & & \multicolumn{2}{c}{\textbf{Dollar volume}} & \multicolumn{2}{c}{\textbf{Stale (\%)}} & \multicolumn{2}{c}{\textbf{Extreme (\%)}} & \multicolumn{2}{c}{\textbf{CS (bps)}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
\textbf{ETF} & \textbf{Index} & \textbf{Early} & \textbf{Late} & \textbf{Early} & \textbf{Late} & \textbf{Early} & \textbf{Late} & \textbf{Early} & \textbf{Late} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# ================================================================== Table 2: sister correlations
rows = []
for seg in SEG:
    a, b = SP[(seg, "R")], SP[(seg, "S")]
    rows.append(dict(seg=seg,
                     sdnA=a.on.std(), sdnB=b.on.std(), sddA=a.id.std(), sddB=b.id.std(),
                     rn=a.on.corr(b.on), rd=a.id.corr(b.id), rc=a.cc.corr(b.cc),
                     rn_e=a.on[:"2007"].corr(b.on[:"2007"]), rn_l=a.on["2020":].corr(b.on["2020":]),
                     rd_e=a.id[:"2007"].corr(b.id[:"2007"]), rd_l=a.id["2020":].corr(b.id["2020":])))
T2 = pd.DataFrame(rows)
T2.to_csv(RES / "sister_corr.csv", index=False)
lines = [f"{NAME[x.seg][0]} / {NAME[x.seg][1]} & {f(x.sdnA,1)} & {f(x.sdnB,1)} & {f(x.sddA,1)} & {f(x.sddB,1)} & "
         f"{f(x.rn)} & {f(x.rd)} & {f(x.rc)} & {f(x.rn_e)} & {f(x.rd_e)} & {f(x.rn_l)} & {f(x.rd_l)} \\\\" for _, x in T2.iterrows()]
write("tab_sister", r"""\begin{table}[H]
\caption{Session spreads and their sister-index correlations. The value-minus-growth spread is computed separately for the overnight (On) and intraday (Id) sessions. Standard deviations are daily, in basis points, for the Russell (R) and S\&P (S) spreads. Correlations are between the Russell spread and the S\&P spread of the same size segment and session; CC is the close-to-close spread. Full sample: 4 September 2001 to 25 September 2026 (6302 days); early: 2001--2007; late: 2020--2026.\label{tab:sister}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{lCCCCCCCCCCC}
\toprule
 & \multicolumn{4}{c}{\textbf{Std. dev. (bps)}} & \multicolumn{3}{c}{\textbf{Sister corr., full}} & \multicolumn{2}{c}{\textbf{Early}} & \multicolumn{2}{c}{\textbf{Late}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
\textbf{Segment} & \textbf{On R} & \textbf{On S} & \textbf{Id R} & \textbf{Id S} & \textbf{On} & \textbf{Id} & \textbf{CC} & \textbf{On} & \textbf{Id} & \textbf{On} & \textbf{Id}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")


# ================================================================== Table 3: own vs cross
def own_cross(own: pd.DataFrame, sis: pd.DataFrame) -> dict:
    n, d, ds, ns = own.on, own.id, sis.id, sis.on
    co, to = cov_t(n, d)
    cx, tx = cov_t(n, ds)
    noise, tn = cov_t(n, ds - d)  # = var(e) estimate (sign: cov(n, d_s - d))
    ols = ols_hac(d, n)
    biv, tiv = iv_slope(d, n, ns)
    return dict(rho_own=n.corr(d), rho_x=n.corr(ds), cov_own=co, t_own=to, cov_x=cx, t_x=tx,
                noise=noise, t_noise=tn, share=noise / -co if co < 0 else np.nan,
                b_ols=ols.params.iloc[1], t_ols=ols.tvalues.iloc[1], b_iv=biv, t_iv=tiv,
                var_on=n.var(), noise_on=noise / n.var())


rows = []
for seg, o, s in PAIRS:
    x = own_cross(SP[(seg, o)], SP[(seg, s)])
    x.update(seg=seg, own=o, label=label(seg, o), sister=label(seg, s))
    rows.append(x)
T3 = pd.DataFrame(rows)
T3.to_csv(RES / "own_cross.csv", index=False)
lines = []
for _, x in T3.iterrows():
    lines.append(f"{x.label} & {x.sister} & {f(x.rho_own)} & {f(x.rho_x)} & {f(x.cov_own,0)}{star(x.t_own)} & {f(x.cov_x,0)}{star(x.t_x)} & "
                 f"{f(x.b_ols)}{star(x.t_ols)} & {f(x.b_iv)}{star(x.t_iv)} & {f(x.noise,0)}{star(x.t_noise)} & {f(100*x.noise_on,0)} \\\\")
write("tab_owncross", r"""\begin{table}[H]
\caption{Own-fund and sister-fund session reversal. For each value-minus-growth spread, own measures relate its overnight spread $n_A$ to its own intraday spread $d_A$, and cross measures relate $n_A$ to the intraday spread $d_B$ of the sister index. $\rho$ is the correlation and Cov the covariance in squared basis points. OLS is the slope of $d_A$ on $n_A$; IV is the same slope with $n_A$ instrumented by the sister overnight spread $n_B$. $\hat\sigma^2_e = \operatorname{Cov}(n_A, d_B - d_A)$ estimates the variance of the fund-specific opening-price error (Equation~(\ref{eq:noise})); the last column expresses it as a percentage of the variance of $n_A$. Significance from Newey--West ($10$ lags) $t$-statistics: $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%. Sample: 4 September 2001 to 25 September 2026.\label{tab:owncross}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{llCCCCCCCC}
\toprule
 & & \multicolumn{2}{c}{$\boldsymbol{\rho}$} & \multicolumn{2}{c}{\textbf{Cov (bps$^2$)}} & \multicolumn{2}{c}{\textbf{Slope}} & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Spread A} & \textbf{Sister B} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{OLS} & \textbf{IV} & $\boldsymbol{\hat\sigma^2_e}$ & \textbf{\% of Var($n_A$)} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# ================================================================== Table 4: subperiods
rows = []
for seg, o, s in PAIRS:
    a, b = SP[(seg, o)], SP[(seg, s)]
    rec = dict(seg=seg, own=o, label=label(seg, o))
    for nm, y0, y1 in SUBS:
        m = (IDX.year >= y0) & (IDX.year <= y1)
        aa, bb = a[m], b[m]
        rec[nm + "_own"] = aa.on.corr(aa.id)
        rec[nm + "_x"] = aa.on.corr(bb.id)
        rec[nm + "_t_own"] = cov_t(aa.on, aa.id)[1]
        rec[nm + "_t_x"] = cov_t(aa.on, bb.id)[1]
        rec[nm + "_nn"] = aa.on.corr(bb.on)
    rows.append(rec)
T4 = pd.DataFrame(rows)
T4.to_csv(RES / "subperiods.csv", index=False)
lines = []
for _, x in T4.iterrows():
    cells = []
    for nm, _, _ in SUBS:
        cells += [f(x[nm + "_own"]) + star(x[nm + "_t_own"]), f(x[nm + "_x"]) + star(x[nm + "_t_x"])]
    lines.append(f"{x.label} & " + " & ".join(cells) + " \\\\")
lines.append("\\midrule")
cells = []
for nm, _, _ in SUBS:
    cells += ["\\multicolumn{2}{c}{" + f(T4[nm + "_nn"].iloc[::2].mean()) + "}"]
lines.append("Sister corr.\\ of $n$ (mean) & " + " & ".join(cells) + " \\\\")
hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{nm}}}}}" for nm, _, _ in SUBS)
cm = "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(4))
write("tab_sub", r"""\begin{table}[H]
\caption{Own and cross session correlations by subperiod. Own is $\rho(n_A, d_A)$; cross is $\rho(n_A, d_B)$, where $B$ is the sister index. The last row averages, across the three size segments, the correlation between the Russell and S\&P overnight spreads. Significance of the corresponding covariance from Newey--West $t$-statistics: $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%.\label{tab:sub}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{lCCCCCCCC}
\toprule
 & """ + hdr + r"""\\
""" + cm + r"""
\textbf{Spread A} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")


# ================================================================== Table 5: print-quality conditioning
def quality_flags(seg, fam):
    g, v, _ = SEG[seg][0 if fam == "R" else 1]
    q = pd.DataFrame(index=IDX)
    q["stale"] = ((R[g].stale + R[v].stale) > 0).astype(float)
    q["ext"] = ((R[g].ext + R[v].ext) > 0).astype(float)
    rv = pd.concat([R[g].relvol, R[v].relvol], axis=1).min(axis=1)
    q["lowvol"] = (rv < 0.5).astype(float)
    csm = (R[g].cs + R[v].cs) / 2
    q["widecs"] = (csm > csm.rolling(250, min_periods=60).quantile(0.8).shift()).astype(float)
    return q


panel = []
for k, (seg, o, s) in enumerate(PAIRS):
    a, b = SP[(seg, o)], SP[(seg, s)]
    q = quality_flags(seg, o)
    df = pd.DataFrame({"pair": k, "n": a.on, "d": a.id, "dS": b.id, "nS": b.on}, index=IDX).join(q)
    df["date"] = np.arange(len(df))
    panel.append(df)
P = pd.concat(panel).dropna().reset_index(drop=True)
rows = []
for flag, nm in [("stale", "Stale open"), ("ext", "Open at high/low"), ("lowvol", "Low volume"), ("widecs", "Wide CS spread")]:
    rec = dict(flag=nm, freq=P[flag].mean() * 100)
    for y, tag in [("d", "own"), ("dS", "cross")]:
        X = pd.DataFrame({"n": P.n, "nQ": P.n * P[flag], "Q": P[flag]})
        X = X.join(pd.get_dummies(P.pair, prefix="p", drop_first=True, dtype=float))
        m = sm.OLS(P[y], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": P.date})
        rec[tag + "_b"], rec[tag + "_tb"] = m.params["n"], m.tvalues["n"]
        rec[tag + "_c"], rec[tag + "_tc"] = m.params["nQ"], m.tvalues["nQ"]
    rows.append(rec)
T5 = pd.DataFrame(rows)
T5.to_csv(RES / "quality.csv", index=False)
lines = [f"{x.flag} & {f(x.freq,1)} & {f(x.own_b,3)}{star(x.own_tb)} & {f(x.own_c,3)}{star(x.own_tc)} & "
         f"{f(x.cross_b,3)}{star(x.cross_tb)} & {f(x.cross_c,3)}{star(x.cross_tc)} \\\\" for _, x in T5.iterrows()]
lines2 = [f" & & ({f(x.own_tb)}) & ({f(x.own_tc)}) & ({f(x.cross_tb)}) & ({f(x.cross_tc)}) \\\\" for _, x in T5.iterrows()]
body = "\n".join(a + "\n" + b for a, b in zip(lines, lines2))
write("tab_quality", r"""\begin{table}[H]
\caption{Reversal conditional on opening-print quality. Pooled regressions over the six spreads of $y_t = a_k + b\,n_{A,t} + c\,n_{A,t} Q_{A,t} + g\,Q_{A,t} + u_t$, where $Q_{A,t}$ indicates low print quality for either fund of spread $A$ and $a_k$ are spread fixed effects. Own: $y = d_A$; cross: $y = d_B$, the sister spread's intraday return. Stale: the open equals the previous close. Open at high/low: the open equals the day's high or low. Low volume: daily volume below half its trailing 60-day median. Wide CS spread: the Corwin--Schultz estimate above its trailing one-year 80th percentile. Freq.\ is the percentage of spread-days flagged. $t$-statistics clustered by date in parentheses; $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%.\label{tab:quality}}
\centering
\small
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
 & & \multicolumn{2}{c}{\textbf{Own ($y=d_A$)}} & \multicolumn{2}{c}{\textbf{Cross ($y=d_B$)}}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Flag $Q$} & \textbf{Freq. (\%)} & $\boldsymbol{b}$ & $\boldsymbol{c}$ & $\boldsymbol{b}$ & $\boldsymbol{c}$\\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

# ================================================================== Table 6: liquidity panel (annual)
rows = []
for k, (seg, o, s) in enumerate(PAIRS):
    a, b = SP[(seg, o)], SP[(seg, s)]
    g, v, _ = SEG[seg][0 if o == "R" else 1]
    for y in range(2002, 2026):
        m = IDX.year == y
        aa, bb = a[m], b[m]
        noise = cov_t(aa.on, bb.id - aa.id)[0]
        csm = ((R[g].cs[m] + R[v].cs[m]) / 2).mean() * BP
        arv = ((R[g].ar[m].mean() + R[v].ar[m].mean()) / 2)
        ar = np.sqrt(max(4 * arv, 0)) * BP
        dv = np.log((R[g].dvol[m].mean() + R[v].dvol[m].mean()) / 2)
        st = ((R[g].stale[m] + R[v].stale[m]) / 2).mean() * 100
        rows.append(dict(pair=k, seg=seg, own=o, year=y, noise=noise, rho_own=aa.on.corr(aa.id),
                         rho_x=aa.on.corr(bb.id), cs=csm, ar=ar, ldv=dv, stale=st,
                         var_on=aa.on.var(), noise_share=noise / aa.on.var()))
A = pd.DataFrame(rows)
A.to_csv(RES / "annual_panel.csv", index=False)
specs = [("cs", "CS spread (bps)"), ("ar", "AR spread (bps)"), ("ldv", "log dollar volume"), ("stale", "Stale opens (\\%)")]
res6 = []
for ycol, ylab in [("noise_share", "Noise share"), ("rho_own", "Own $\\rho$"), ("rho_x", "Cross $\\rho$")]:
    for xcol, xlab in specs:
        X = pd.DataFrame({xcol: A[xcol]}).join(pd.get_dummies(A.pair, prefix="p", drop_first=True, dtype=float))
        X["trend"] = A.year - 2002
        m = sm.OLS(A[ycol], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": A.year})
        m0 = sm.OLS(A[ycol], sm.add_constant(X.drop(columns="trend"))).fit(cov_type="cluster", cov_kwds={"groups": A.year})
        res6.append(dict(y=ylab, x=xlab, b0=m0.params[xcol], t0=m0.tvalues[xcol], r20=m0.rsquared,
                         b1=m.params[xcol], t1=m.tvalues[xcol], r21=m.rsquared))
T6 = pd.DataFrame(res6)
T6.to_csv(RES / "liquidity_panel.csv", index=False)
lines = []
for ylab in T6.y.unique():
    sub = T6[T6.y == ylab]
    first = True
    for _, x in sub.iterrows():
        lines.append(f"{ylab if first else ''} & {x.x} & {f(x.b0,3)}{star(x.t0)} & ({f(x.t0)}) & {f(x.r20)} & {f(x.b1,3)}{star(x.t1)} & ({f(x.t1)}) & {f(x.r21)} \\\\")
        first = False
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_liq", r"""\begin{table}[H]
\caption{Opening-print noise and liquidity: annual panel. Each observation is one of the six spreads in one calendar year, 2002--2025 ($n = 144$). Noise share is $\hat\sigma^2_e$ divided by the variance of the overnight spread; own and cross $\rho$ are as in Table~\ref{tab:owncross}. Regressors are the average of the two style funds' Corwin--Schultz (CS) or Abdi--Ranaldo (AR) spread estimates, the log of their average daily dollar volume and their percentage of stale opens. All regressions include spread fixed effects; the right panel adds a linear time trend. $t$-statistics clustered by year in parentheses; $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%.\label{tab:liq}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{llCCCCCC}
\toprule
 & & \multicolumn{3}{c}{\textbf{Fixed effects}} & \multicolumn{3}{c}{\textbf{Fixed effects + trend}}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
\textbf{Dependent} & \textbf{Regressor} & \textbf{Coef.} & $\boldsymbol{t}$ & $\boldsymbol{R^2}$ & \textbf{Coef.} & $\boldsymbol{t}$ & $\boldsymbol{R^2}$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")


# ================================================================== Table 7: risk measurement
def risk_row(n, d, ds, cc):
    s2e = cov_t(n, ds - d)[0]
    vn, vd, c = n.var(), d.var(), n.cov(d)
    vn_c, vd_c, c_c = vn - s2e, vd - s2e, c + s2e
    vcc = cc.var()
    return dict(s2e=s2e, vol_on=np.sqrt(vn * 252), vol_on_c=np.sqrt(max(vn_c, 0) * 252),
                vol_id=np.sqrt(vd * 252), vol_id_c=np.sqrt(max(vd_c, 0) * 252),
                share_on=vn / vcc, share_on_c=vn_c / vcc, rho=c / np.sqrt(vn * vd),
                rho_c=c_c / np.sqrt(max(vn_c, 1e-12) * max(vd_c, 1e-12)), vol_cc=np.sqrt(vcc * 252))


rows = []
for seg, o, s in PAIRS:
    a, b = SP[(seg, o)], SP[(seg, s)]
    x = risk_row(a.on, a.id, b.id, a.cc)
    x.update(kind="Spread", name=f"{label(seg, o)} V$-$G", seg=seg)
    rows.append(x)
for seg, (ru, sp) in SEG.items():
    for t in ru + sp:
        x = risk_row(R[t].on * BP, R[t].id * BP, R[SISTER[t]].id * BP, R[t].cc * BP)
        x.update(kind="ETF", name=t, seg=seg)
        rows.append(x)
T7 = pd.DataFrame(rows)
T7.to_csv(RES / "risk.csv", index=False)
lines = []
for kind in ("Spread", "ETF"):
    for _, x in T7[T7.kind == kind].iterrows():
        lines.append(f"{x['name']} & {f(x.vol_cc/100,2)} & {f(x.vol_on/100,2)} & {f(x.vol_on_c/100,2)} & {f(x.vol_id/100,2)} & {f(x.vol_id_c/100,2)} & "
                     f"{f(100*x.share_on,0)} & {f(100*x.share_on_c,0)} & {f(x.rho)} & {f(x.rho_c)} \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_risk", r"""\begin{table}[H]
\caption{Session risk before and after removing opening-print noise. Volatilities are annualized, in percent; raw values use reported prices, and corrected values subtract the estimated noise variance $\hat\sigma^2_e$ from the overnight and intraday variances and add it to their covariance (Equation~(\ref{eq:correct})). On share is the overnight variance as a percentage of the close-to-close variance; $\rho$ is the overnight--intraday correlation. For individual ETFs, $\hat\sigma^2_e$ uses the sister fund tracking the same size and style segment of the other index family. Sample: 4 September 2001 to 25 September 2026.\label{tab:risk}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{lCCCCCCCCC}
\toprule
 & & \multicolumn{2}{c}{\textbf{Overnight vol.}} & \multicolumn{2}{c}{\textbf{Intraday vol.}} & \multicolumn{2}{c}{\textbf{On share (\%)}} & \multicolumn{2}{c}{$\boldsymbol{\rho}$\textbf{(on, id)}}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
\textbf{Series} & \textbf{CC vol.} & \textbf{Raw} & \textbf{Corr.} & \textbf{Raw} & \textbf{Corr.} & \textbf{Raw} & \textbf{Corr.} & \textbf{Raw} & \textbf{Corr.}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# ================================================================== Table 8: closing boundary
rows = []
for seg, o, s in PAIRS:
    a, b = SP[(seg, o)], SP[(seg, s)]
    # close noise: enters d_t (+) and n_{t+1} (-)
    co = cov_t(a.id, a.on.shift(-1)); cx = cov_t(a.id, b.on.shift(-1))
    ro, rx = a.id.corr(a.on.shift(-1)), a.id.corr(b.on.shift(-1))
    ao = cov_t(a.cc, a.cc.shift(-1)); ax = cov_t(a.cc, b.cc.shift(-1))
    rows.append(dict(label=label(seg, o), r_dn_own=ro, t_dn_own=co[1], r_dn_x=rx, t_dn_x=cx[1],
                     s2c=cov_t(a.id, b.on.shift(-1) - a.on.shift(-1))[0], t_s2c=cov_t(a.id, b.on.shift(-1) - a.on.shift(-1))[1],
                     s2o=cov_t(a.on, b.id - a.id)[0],
                     ac_own=a.cc.corr(a.cc.shift(-1)), t_ac_own=ao[1], ac_x=a.cc.corr(b.cc.shift(-1)), t_ac_x=ax[1]))
T8 = pd.DataFrame(rows)


T8.to_csv(RES / "closing.csv", index=False)
lines = [f"{x.label} & {f(x.r_dn_own)}{star(x.t_dn_own)} & {f(x.r_dn_x)}{star(x.t_dn_x)} & {f(x.s2c,0)}{star(x.t_s2c)} & {f(x.s2o,0)} & {f(x.ac_own)}{star(x.t_ac_own)} & {f(x.ac_x)}{star(x.t_ac_x)} \\\\"
         for _, x in T8.iterrows()]
write("tab_close", r"""\begin{table}[H]
\caption{The closing boundary. Intraday-to-next-overnight: own is $\rho(d_{A,t}, n_{A,t+1})$ and cross is $\rho(d_{A,t}, n_{B,t+1})$; a fund-specific error in the closing price enters $d_{A,t}$ and $n_{A,t+1}$ with opposite signs, and $\hat\sigma^2_c = \operatorname{Cov}(d_{A,t}, n_{B,t+1} - n_{A,t+1})$ estimates its variance in squared basis points, for comparison with the opening-price estimate $\hat\sigma^2_e$ of Table~\ref{tab:owncross}. Close-to-close: own is the first-order autocorrelation of spread $A$ and cross is $\rho(s_{A,t}, s_{B,t+1})$. The bottom panel replaces the Russell sister with the SPDR S\&P 500 style ETFs (SPYG, SPYV), which track the same indices as IVW and IVE over 2018--2025 (adjusted closes only). Significance of the corresponding covariance from Newey--West $t$-statistics: $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%.\label{tab:close}}
\centering
\small
\begin{tabularx}{\textwidth}{lCCCCCC}
\toprule
 & \multicolumn{4}{c}{\textbf{Intraday $\to$ next overnight}} & \multicolumn{2}{c}{\textbf{Close-to-close}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
\textbf{Spread A} & \textbf{Own} & \textbf{Cross} & $\boldsymbol{\hat\sigma^2_c}$ & $\boldsymbol{\hat\sigma^2_e}$ & \textbf{Own} & \textbf{Cross}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")


# ================================================================== Table 9: paper profits own vs sister signal
def ls_perf(sig: pd.Series, ret: pd.Series) -> tuple[float, float, float]:
    pos = -np.sign(sig)
    pnl = (pos * ret).dropna()
    return pnl.mean() * 252 / 100, pnl.mean() / pnl.std() * np.sqrt(252), pnl.mean() / np.sqrt(nw_var_mean(pnl.values))


rows = []
for seg, o, s in PAIRS:
    a, b = SP[(seg, o)], SP[(seg, s)]
    rec = dict(label=label(seg, o))
    for per, m in [("full", np.ones(len(IDX), bool)), ("late", IDX.year >= 2020)]:
        mo = ls_perf(a.on[m], a.id[m]); mx = ls_perf(b.on[m], a.id[m])
        rec.update({f"{per}_own_mu": mo[0], f"{per}_own_sr": mo[1], f"{per}_own_t": mo[2],
                    f"{per}_x_mu": mx[0], f"{per}_x_sr": mx[1], f"{per}_x_t": mx[2]})
    rows.append(rec)
T9 = pd.DataFrame(rows)
T9.to_csv(RES / "paper_profits.csv", index=False)
lines = [f"{x.label} & {f(x.full_own_mu,1)} & {f(x.full_own_sr)}{star(x.full_own_t)} & {f(x.full_x_mu,1)} & {f(x.full_x_sr)}{star(x.full_x_t)} & "
         f"{f(x.late_own_mu,1)} & {f(x.late_own_sr)}{star(x.late_own_t)} & {f(x.late_x_mu,1)} & {f(x.late_x_sr)}{star(x.late_x_t)} \\\\" for _, x in T9.iterrows()]
write("tab_profits", r"""\begin{table}[H]
\caption{Gross intraday reversal returns with own and sister signals. The intraday position in spread $A$ is short value and long growth (long value and short growth) when the overnight signal is positive (negative): $\pi_t = -\operatorname{sign}(x_t)\, d_{A,t}$. Own uses $x = n_A$; sister uses $x = n_B$. Mean is the annualized gross return in percent of the notional of one leg, before any costs; SR is the annualized Sharpe ratio. Both signals use the opening price and are diagnostic rather than executable. Significance of the mean from Newey--West $t$-statistics: $^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%.\label{tab:profits}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{lCCCCCCCC}
\toprule
 & \multicolumn{4}{c}{\textbf{2001--2026}} & \multicolumn{4}{c}{\textbf{2020--2026}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
 & \multicolumn{2}{c}{\textbf{Own signal}} & \multicolumn{2}{c}{\textbf{Sister signal}} & \multicolumn{2}{c}{\textbf{Own signal}} & \multicolumn{2}{c}{\textbf{Sister signal}}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Spread A} & \textbf{Mean} & \textbf{SR} & \textbf{Mean} & \textbf{SR} & \textbf{Mean} & \textbf{SR} & \textbf{Mean} & \textbf{SR}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# ================================================================== Table 10: robustness
rows = []
for seg, o, s in PAIRS:
    a, b = SP[(seg, o)].copy(), SP[(seg, s)].copy()
    g, v, _ = SEG[seg][0 if o == "R" else 1]
    gs, vs, _ = SEG[seg][1 if o == "R" else 0]
    rec = dict(label=label(seg, o))
    # (i) drop days with any stale open in the four funds
    m = (R[g].stale + R[v].stale + R[gs].stale + R[vs].stale) == 0
    rec["nostale_own"], rec["nostale_x"] = a.on[m].corr(a.id[m]), a.on[m].corr(b.id[m])
    # (ii) log returns
    la = {ss: np.log1p(R[v][ss]) - np.log1p(R[g][ss]) for ss in ("on", "id")}
    lb = {ss: np.log1p(R[vs][ss]) - np.log1p(R[gs][ss]) for ss in ("on", "id")}
    rec["log_own"], rec["log_x"] = la["on"].corr(la["id"]), la["on"].corr(lb["id"])
    # (iii) winsorize at 0.5%/99.5%
    w = lambda x: x.clip(x.quantile(0.005), x.quantile(0.995))
    rec["win_own"], rec["win_x"] = w(a.on).corr(w(a.id)), w(a.on).corr(w(b.id))
    # (iv) Spearman
    rec["sp_own"], rec["sp_x"] = a.on.corr(a.id, method="spearman"), a.on.corr(b.id, method="spearman")
    # (v) excluding crisis windows
    ex = ~(((IDX >= "2008-09-01") & (IDX <= "2009-06-30")) | ((IDX >= "2020-02-15") & (IDX <= "2020-06-30")))
    rec["ex_own"], rec["ex_x"] = a.on[ex].corr(a.id[ex]), a.on[ex].corr(b.id[ex])
    rows.append(rec)
T10 = pd.DataFrame(rows)
T10.to_csv(RES / "robust.csv", index=False)
cols = ["nostale", "log", "win", "sp", "ex"]
lines = [f"{x.label} & " + " & ".join(f"{f(x[c+'_own'])} & {f(x[c+'_x'])}" for c in cols) + " \\\\" for _, x in T10.iterrows()]
write("tab_robust", r"""\begin{table}[H]
\caption{Robustness of the own--cross contrast. Each pair of columns reports the own correlation $\rho(n_A, d_A)$ and the cross correlation $\rho(n_A, d_B)$ under one variation: excluding days on which any of the four funds has a stale open; log instead of simple returns; returns winsorized at the 0.5th and 99.5th percentiles; Spearman rank correlation; and excluding September 2008--June 2009 and 15 February--30 June 2020.\label{tab:robust}}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\small
\begin{tabularx}{\fulllength}{lCCCCCCCCCC}
\toprule
 & \multicolumn{2}{c}{\textbf{No stale}} & \multicolumn{2}{c}{\textbf{Log}} & \multicolumn{2}{c}{\textbf{Winsorized}} & \multicolumn{2}{c}{\textbf{Spearman}} & \multicolumn{2}{c}{\textbf{Ex-crises}}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
\textbf{Spread A} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross} & \textbf{Own} & \textbf{Cross}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# ================================================================== 24 August 2015 sensitivity
rows = []
ex15 = IDX != pd.Timestamp("2015-08-24")
for seg, o, s_ in PAIRS:
    a, b = SP[(seg, o)], SP[(seg, s_)]
    rows.append(dict(label=label(seg, o), own=a.on[ex15].corr(a.id[ex15]), x=a.on[ex15].corr(b.id[ex15]),
                     n_0824=a.on.loc["2015-08-24"], d_0824=a.id.loc["2015-08-24"],
                     own_1519_ex=a.on[ex15 & (IDX.year >= 2015) & (IDX.year <= 2019)].corr(a.id[ex15 & (IDX.year >= 2015) & (IDX.year <= 2019)])))
pd.DataFrame(rows).to_csv(RES / "aug2015.csv", index=False)

# ================================================================== benchmarks (single ETFs) summary for text
rows = []
for seg, (ru, sp) in SEG.items():
    for t in ru + sp:
        n, d, ds = R[t].on, R[t].id, R[SISTER[t]].id
        rows.append(dict(t=t, seg=seg, rho_own=n.corr(d), rho_x=n.corr(ds), noise=cov_t(n * BP, (ds - d) * BP)[0],
                         t_noise=cov_t(n * BP, (ds - d) * BP)[1], var_on=(n * BP).var()))
pd.DataFrame(rows).to_csv(RES / "single_etf.csv", index=False)

# ================================================================== figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C_OWN, C_X = "#1f4e79", "#c55a11"

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, seg in zip(axs, SEG):
    for fam, ls in (("R", "-"), ("S", "--")):
        a, b = SP[(seg, fam)], SP[(seg, "S" if fam == "R" else "R")]
        own = a.on.rolling(252).corr(a.id)
        x = a.on.rolling(252).corr(b.id)
        ax.plot(own.index, own, ls, color=C_OWN, lw=1, label=f"Own, {label(seg, fam).replace(chr(92), '')}")
        ax.plot(x.index, x, ls, color=C_X, lw=1, label=f"Cross, {label(seg, fam).replace(chr(92), '')}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(seg + " capitalization")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
axs[0].set_ylabel("Rolling 252-day correlation")
fig.tight_layout()
fig.savefig(FIG / "fig_rolling.pdf")
fig.savefig(FIG / "fig_rolling.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
for ax, seg in zip(axs, SEG):
    a, b = SP[(seg, "R")], SP[(seg, "S")]
    yrs = sorted(set(IDX.year))
    rn = [a.on[IDX.year == y].corr(b.on[IDX.year == y]) for y in yrs]
    rd = [a.id[IDX.year == y].corr(b.id[IDX.year == y]) for y in yrs]
    ax.plot(yrs, rd, "-o", ms=3, color="#555555", lw=1, label="Intraday spreads")
    ax.plot(yrs, rn, "-s", ms=3, color=C_X, lw=1, label="Overnight spreads")
    ax.set_title(f"{NAME[seg][0]} vs {NAME[seg][1]}".replace("\\", ""))
    ax.set_ylim(-0.45, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
axs[0].set_ylabel("Russell--S&P correlation, by year")
fig.tight_layout()
fig.savefig(FIG / "fig_sister.pdf")
fig.savefig(FIG / "fig_sister.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
mk = {"Large": "o", "Mid": "s", "Small": "^"}
for seg in SEG:
    for fam, col in (("R", C_OWN), ("S", C_X)):
        s = A[(A.seg == seg) & (A.own == fam)]
        ax.scatter(np.exp(s.ldv) / 1e6, 100 * s.noise_share, marker=mk[seg], s=14, color=col, alpha=0.75,
                   label=f"{label(seg, fam)}".replace("\\", ""))
ax.set_xscale("log")
ax.set_xlabel("Average daily dollar volume of the two style ETFs (USD million, log scale)")
ax.set_ylabel("Noise share of overnight variance (%)")
ax.axhline(0, color="grey", lw=0.6)
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(FIG / "fig_liq.pdf")
fig.savefig(FIG / "fig_liq.png", dpi=150)
plt.close(fig)

print("done", len(IDX), IDX[0].date(), IDX[-1].date())
