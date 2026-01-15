import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from crewai.tools import tool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns: (field, ticker).
    Flatten into single-level columns like 'Close' / 'Open' etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the first level (field name)
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _file_summary_csv(path: str | Path) -> dict:
    p = Path(path)
    summary = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return summary
    summary["bytes"] = p.stat().st_size
    try:
        cols = list(pd.read_csv(p, nrows=1).columns)
        summary["columns"] = cols
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            summary["rows"] = max(sum(1 for _ in f) - 1, 0)
    except Exception as e:
        summary["inspect_error"] = str(e)
    return summary


@tool("yahoo_download_ohlcv_1d_tool")
def yahoo_download_ohlcv_1d_tool(
    topic: str,
    ticker: str,
    start_date: str,
    end_date: str,
    out_csv_path: str = "",
    write_metadata: bool = True,
    metadata_path: str = "",
) -> str:
    """
    Download daily OHLCV from Yahoo Finance for a given ticker and write to CSV.
    Robust against MultiIndex columns and minor schema variations.
    """
    if not out_csv_path:
        out_csv_path = f"data/raw/{topic}_yahoo_ohlcv_1d.csv"
    if not metadata_path:
        metadata_path = f"data/raw/{topic}_extraction_metadata.json"

    out_csv_path = str(Path(out_csv_path))
    metadata_path = str(Path(metadata_path))

    _ensure_dir(Path(out_csv_path).parent)
    _ensure_dir(Path(metadata_path).parent)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned from Yahoo Finance for ticker={ticker} in {start_date}..{end_date}. "
            "Possible causes: invalid ticker, rate limiting, or network blocks."
        )

    df = _flatten_columns(df)
    df = df.reset_index()

    # Normalize date column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    elif "date" not in df.columns:
        raise ValueError(f"Yahoo response missing date column. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    # Normalize OHLCV names (accept a couple common variants)
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_map)

    # Ensure we have a usable price column
    if "close" not in df.columns:
        # Fallback: if only adj_close exists, use it as close
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        else:
            raise ValueError(
                "Yahoo response did not include Close or Adj Close after normalization. "
                f"Columns={list(df.columns)}"
            )

    # Keep expected fields if present
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["commodity"] = topic
    df["ticker"] = ticker

    # Coerce numerics
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing canonical price
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    df.to_csv(out_csv_path, index=False, na_rep="")

    if write_metadata:
        meta = {
            "as_of_utc": _utc_now_iso(),
            "provider": "yahoo_finance",
            "topic": topic,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "schema": "ohlcv-1d",
            "outputs": {
                "ohlcv_csv": _file_summary_csv(out_csv_path),
            },
            "notes": "Yahoo Finance OHLCV is used as a free price proxy; may not match official exchange settlement.",
        }
        Path(metadata_path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return out_csv_path
