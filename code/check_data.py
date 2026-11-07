"""Compare a data directory with the files used in the paper.

    python code/check_data.py                 # verify data/ against data/MANIFEST.csv
    python code/check_data.py --dir data_new  # check a fresh download
    python code/check_data.py --write         # regenerate MANIFEST.csv (maintainers only)

For each file the manifest records the SHA-256 hash, row count, first and last
date. A hash mismatch is expected after a fresh Yahoo download, because Yahoo
revises dividend adjustments; the report then shows how far the session returns
used in the paper moved, which is what matters for replication.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "MANIFEST.csv"
START, END = "2001-09-04", "2026-09-25"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, index_col=0)
    d = d[pd.to_datetime(d.index, errors="coerce").notna()]
    d.index = pd.to_datetime(d.index)
    return d.apply(pd.to_numeric, errors="coerce").sort_index()


def files(base: Path) -> list[Path]:
    return sorted(base.glob("*.csv"))


def write_manifest(base: Path) -> None:
    rows = []
    for p in files(base):
        d = read(p)
        rows.append(dict(file=str(p.relative_to(base)), sha256=sha256(p), rows=len(d),
                         first=d.index[0].date(), last=d.index[-1].date()))
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    (base / "CHECKSUMS.sha256").write_text("".join(f"{r['sha256']}  {r['file']}\n" for r in rows))
    print(f"wrote {MANIFEST} ({len(rows)} files)")


def session_returns(d: pd.DataFrame) -> pd.DataFrame:
    adj = d["Adj Close"] if "Adj Close" in d else d["Close"]
    r = pd.DataFrame({"cc": adj.pct_change()})
    if {"Open", "Close"} <= set(d.columns):
        r["id"] = d["Close"] / d["Open"] - 1
        r["on"] = (1 + r.cc) / (1 + r.id) - 1
    return r.loc[START:END]


def check(base: Path) -> None:
    man = pd.read_csv(MANIFEST)
    ref_base = ROOT / "data"
    ok = same_returns = 0
    for _, m in man.iterrows():
        p = base / m.file
        if not p.exists():
            print(f"MISSING  {m.file}")
            continue
        if sha256(p) == m.sha256:
            ok += 1
            continue
        d = read(p)
        msg = f"DIFFERS  {m.file}: rows {len(d)} vs {m.rows}, {d.index[0].date()}..{d.index[-1].date()}"
        ref = ref_base / m.file
        if ref.exists() and ref.resolve() != p.resolve():
            a, b = session_returns(read(ref)), session_returns(d)
            j = a.join(b, rsuffix="_new", how="inner")
            diffs = {c: np.nanmax(np.abs(j[c] - j[c + "_new"])) * 1e4 for c in a.columns}
            msg += "; max |diff| in bps: " + ", ".join(f"{k}={v:.3f}" for k, v in diffs.items())
            if max(diffs.values()) < 1e-3:
                same_returns += 1
                msg += "  (returns identical)"
        print(msg)
    print(f"{ok} of {len(man)} files byte-identical to the paper's data; "
          f"{same_returns} more differ only in formatting (session returns identical to 0.001 bps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        write_manifest(Path(a.dir))
    else:
        check(Path(a.dir))


if __name__ == "__main__":
    main()
