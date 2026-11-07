# Noise or News at the Opening Bell?

Replication code and data for

> Salotra, G. and Pinsky, E. (2026). *Noise or News at the Opening Bell? Fund-Specific Pricing Errors and the Overnight–Intraday Reversal in Growth–Value ETF Spreads.* Submitted to the *Journal of Risk and Financial Management*.

The value-minus-growth spread of US style ETFs often reverses sharply between the overnight and the following intraday session. The paper shows that this reversal is fund-specific error in opening prices, not style-level return dynamics. It uses two designs:

1. **Sister indices, 2001–2026.** Each iShares Russell style spread is compared with the iShares S&P style spread covering the same size segment.
2. **Same index, 2011–2026.** iShares, Vanguard and SPDR funds that track the *identical* style index are compared with each other. This design also measures each fund's own opening and closing noise and relates it to liquidity, and it evaluates the consequences for overnight risk measures (variance forecasts, VaR).

## Reproduce the results

```bash
pip install -r requirements.txt
python code/analysis.py      # long sample, Russell vs S&P sisters
python code/same_index.py    # same-index funds, fund-level noise, risk tests
```

Each script takes about 10–20 seconds and writes to `results/`, `tables/` and `figures/`. The repository already contains all outputs, so you can check numbers without running anything.

## Contents

| Path | Description |
|---|---|
| `code/analysis.py` | Long-sample tests (2001–2026): sister correlations, own vs. cross reversal, IV, noise variance, subperiods, print quality, liquidity panel, closing boundary, session risk, paper profits, robustness |
| `code/same_index.py` | Same-index tests (2011–2026): own vs. same-index cross reversal, three-fund decomposition of fund-level noise, liquidity within index and year, risk correction, variance forecasts, VaR, paper profits, robustness |
| `code/download_data.py` | Re-downloads all 41 price series from Yahoo Finance |
| `code/check_data.py` | Compares a data folder with the files used in the paper |
| `data/` | Daily prices for the 41 funds (see *Data* below) |
| `data/MANIFEST.csv`, `data/CHECKSUMS.sha256` | SHA-256 hash, row count and date range of each data file |
| `results/` | Full-precision CSVs behind every number in the text |
| `tables/` | LaTeX tables, as input by the manuscript |
| `figures/` | Figures (PDF for the manuscript, PNG for viewing) |

## Map from the paper to the outputs

Table numbers refer to the submitted manuscript.

| Paper | LaTeX output | Underlying estimates | Script |
|---|---|---|---|
| Table 1 – iShares funds and print quality | `tab_data.tex` | `data_summary.csv` | analysis |
| Table 2 – Russell/S&P sister correlations | `tab_sister.tex` | `sister_corr.csv` | analysis |
| Table 3 – Own vs. sister reversal, IV, noise | `tab_owncross.tex` | `own_cross.csv` | analysis |
| Table 4 – Subperiods (long sample) | `tab_sub.tex` | `subperiods.csv` | analysis |
| Table 5 – Print quality (long sample) | `tab_quality.tex` | `quality.csv` | analysis |
| Table 6 – Noise and liquidity over time | `tab_liq.tex` | `liquidity_panel.csv`, `annual_panel.csv` | analysis |
| Table 7 – Closing boundary | `tab_close.tex` | `closing.csv` | analysis |
| Table 8 – Same-index sister correlations | `tab_samecorr.tex` | `same_corr.csv` | same_index |
| Table 9 – Own vs. same-index reversal (main test) | `tab_same.tex` | `same_test.csv` | same_index |
| Table 10 – Same-index subperiods | `tab_samesub.tex` | `same_sub.csv` | same_index |
| Table 11 – Print quality (same index) | `tab_samequal.tex` | `same_quality.csv` | same_index |
| Table 12 – Fund-level opening and closing noise | `tab_fundnoise.tex` | `fund_noise.csv` | same_index |
| Table 13 – Fund noise vs. liquidity within index-year | `tab_fundliq.tex` | `fund_liq.csv`, `fund_panel.csv` | same_index |
| Table 14 – Noise-corrected session risk (long sample) | `tab_risk.tex` | `risk.csv`, `single_etf.csv` | analysis |
| Table 15 – Noise-corrected session risk (same index) | `tab_samerisk.tex` | `same_risk.csv` | same_index |
| Table 16 – Overnight variance forecasts | `tab_forecast.tex` | `forecast.csv` | same_index |
| Table 17 – Overnight VaR | `tab_var.tex` | `var.csv` | same_index |
| Table 18 – Paper profits (long sample) | `tab_profits.tex` | `paper_profits.csv` | analysis |
| Table 19 – Paper profits (same index) | `tab_sameprof.tex` | `same_profits.csv` | same_index |
| Table 20 – Robustness (long sample) | `tab_robust.tex` | `robust.csv` | analysis |
| Table 21 – Robustness (same index) | `tab_samerobust.tex` | `same_robust.csv` | same_index |
| Table A1 – Funds in the same-index sample | `tab_funds.tex` | `fund_list.csv`, `family_summary.csv` | same_index |
| Table A2 – All same-index pairs | `tab_pairs.tex` | `same_pairs.csv` | same_index |
| Figure 1 – Russell/S&P correlations by year | `fig_sister.pdf` | — | analysis |
| Figure 2 – Rolling own/cross correlations (long sample) | `fig_rolling.pdf` | — | analysis |
| Figure 3 – Noise share vs. dollar volume | `fig_liq.pdf` | `annual_panel.csv` | analysis |
| Figure 4 – Rolling own/cross correlations (same index) | `fig_samerolling.pdf` | — | same_index |
| Figure 5 – Fund-level opening noise by year | `fig_fundnoise.pdf` | `fund_panel.csv` | same_index |
| Figure 6 – Overnight volatility forecasts | `fig_forecast.pdf` | `forecast.csv` | same_index |
| 24 August 2015 example (Section 5.3) | — | `aug2015.csv` | analysis |

## Data

**Funds.** The data cover 41 funds: 18 iShares, 15 Vanguard and 8 SPDR. Each `data/<TICKER>.csv` has the columns `Date, Open, High, Low, Close, Adj Close, Volume, Dividends`. Open, High, Low and Close are split-adjusted but not dividend-adjusted. `Adj Close` is adjusted for both splits and dividends. All files come from a single Yahoo Finance download, so dividend adjustments are consistent across funds.

| Index | iShares | Vanguard | SPDR |
|---|---|---|---|
| Russell 1000 / Growth / Value | IWB / IWF / IWD | VONE / VONG / VONV | — |
| Russell Midcap / Growth / Value | IWR / IWP / IWS | — | — |
| Russell 2000 / Growth / Value | IWM / IWO / IWN | VTWO / VTWG / VTWV | — |
| S&P 500 / Growth / Value | IVV / IVW / IVE | VOO / VOOG / VOOV | SPY / SPYG / SPYV |
| S&P MidCap 400 / Growth / Value | IJH / IJK / IJJ | IVOO / IVOG / IVOV | MDY / MDYG / MDYV |
| S&P SmallCap 600 / Growth / Value | IJR / IJT / IJS | VIOO / VIOG / VIOV | — / SLYG / SLYV |

**Samples.**
- *Long sample:* 4 September 2001 – 25 September 2026, iShares funds only, 6302 trading days.
- *Same-index sample:* 3 January 2011 – 25 September 2026, excluding 24 August 2015, 3955 trading days. It starts in 2011 because the SPDR style funds tracked other indices before then: their annual returns differ from the iShares funds' by up to 24 percentage points before 2011 and by at most 1.3 points afterwards. The Vanguard funds launched in September 2010.

**Session returns.**
- close-to-close: `cc = AdjClose_t / AdjClose_{t-1} - 1`
- intraday: `id = Close_t / Open_t - 1`
- overnight: `on = (1 + cc) / (1 + id) - 1`

The overnight return is defined residually, so it includes dividends.

**Source and terms.** All prices come from Yahoo Finance and remain subject to Yahoo's terms of use. They are included only so the paper can be reproduced exactly. The MIT license covers the code, not the data.

### Re-downloading the data

```bash
python code/download_data.py --out data_new
python code/check_data.py --dir data_new
```

`download_data.py` calls yfinance with `auto_adjust=False`. **Do not change this setting.** Since yfinance 0.2.51 the default replaces raw opens and closes with dividend-adjusted values and drops `Adj Close`, which silently breaks the overnight/intraday split.

Yahoo revises historical dividend adjustments, so a fresh download will rarely be byte-identical to the files used in the paper. `check_data.py` reports, file by file, the largest change in the session returns, in basis points. For example, a re-download of the iShares files moved session returns by at most 0.02 bps. To run the analysis on a fresh download, replace `data/` with `data_new/`.

## Methods in brief

- **Noise variance of a spread.** Estimated by `Cov(n_A, d_B − d_A)`, where `n` is the overnight spread, `d` the intraday spread and `B` a sister fund (or the average of same-index sisters).
- **Reversal net of noise.** The instrumental-variable slope regresses `d_A` on `n_A`, using the sister overnight spread as the instrument.
- **Fund-level noise (three-cornered hat).** For funds A, B, C on the same index:
  - `S_AB = −Cov(n_A − n_B, d_A − d_B) = σ²_A + σ²_B`
  - `σ²_A = (S_AB + S_AC − S_BC)/2`
  - The closing-price analogue uses the intraday difference at t and the overnight difference at t+1.
- **Variance forecasts.** Evaluated against the proxy `n_A · n̄_B` with squared-error loss and Diebold–Mariano tests.
- **Standard errors.** Newey–West with 10 lags unless the table states otherwise.

See Section 3 of the paper for the measurement model.

## License and citation

Code is under the MIT License (see `LICENSE`). Please cite the paper if you use the code; `CITATION.cff` has the details.
