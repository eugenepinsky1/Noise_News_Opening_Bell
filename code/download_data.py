"""Download the daily price data used in the paper from Yahoo Finance.

    python code/download_data.py                  # paper sample, writes data/
    python code/download_data.py --out data_new   # write elsewhere (e.g. to compare)

41 funds (18 iShares, 15 Vanguard, 8 SPDR), one CSV each with columns
Date, Open, High, Low, Close (split-adjusted, NOT dividend-adjusted),
Adj Close (split- and dividend-adjusted), Volume, Dividends.

IMPORTANT: auto_adjust must be False. Since yfinance 0.2.51 the default is True,
which replaces Open/High/Low/Close with dividend-adjusted values and drops
'Adj Close'. The session decomposition r_id = Close/Open - 1 requires raw
opens and closes, and r_cc requires Adj Close, so the default silently breaks
the overnight/intraday split.

Yahoo revises dividend adjustments retroactively, so a fresh download will not
match the paper's files bit for bit. Run `python code/check_data.py --dir <out>`
to see which files differ and by how much.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ISHARES = ["IWB", "IWF", "IWD", "IWR", "IWP", "IWS", "IWM", "IWO", "IWN",
           "IVV", "IVW", "IVE", "IJH", "IJK", "IJJ", "IJR", "IJT", "IJS"]
VANGUARD = ["VONE", "VONG", "VONV", "VTWO", "VTWG", "VTWV", "VOO", "VOOG", "VOOV",
            "IVOO", "IVOG", "IVOV", "VIOO", "VIOG", "VIOV"]
SPDR = ["SPY", "SPYG", "SPYV", "MDY", "MDYG", "MDYV", "SLYG", "SLYV"]
TICKERS = ISHARES + VANGUARD + SPDR
COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends"]

# Paper sample ends 25 Sep 2026; yfinance treats `end` as exclusive.
END = "2026-09-26"
START = "2000-01-01"


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported here so the module can be inspected without it

    d = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False,
                                  actions=True, repair=False)
    if d.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return format_frame(d)


def format_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Bring a yfinance frame into the layout read by analysis.py."""
    missing = [c for c in COLS if c not in d.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}; was auto_adjust=False used?")
    d = d[COLS].copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d.index.name = "Date"
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["Volume"] = d["Volume"].astype("int64")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--start", default=START)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        d = fetch(t, args.start, END)
        d.to_csv(out / f"{t}.csv")
        print(f"{t}: {d.index[0].date()} to {d.index[-1].date()}, {len(d)} rows")


if __name__ == "__main__":
    main()
