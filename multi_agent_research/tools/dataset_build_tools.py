from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from crewai.tools import tool


def _read_csv_if_exists(path: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def target_next_trading_day_on_or_after(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "front_month_price",
    days: int = 14,
) -> pd.Series:
    """
    Target = price on the first available trading day on/after (date + days calendar days).

    Example:
      if date=2015-01-02 and days=14 => anchor=2015-01-16
      target is price at the first trading date >= 2015-01-16.

    Deterministic: sorts internally but returns aligned to original df index.
    """
    # Work on a minimal copy and keep original row identity
    base = df[[date_col, price_col]].copy()
    base["_row_id"] = np.arange(len(base))

    base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
    base = base.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Build a sorted "future trading calendar" table (unique dates)
    future = base[[date_col, price_col]].dropna(subset=[price_col]).copy()
    future = future.drop_duplicates(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # For each row, compute the anchor date = date + days
    base["_anchor"] = base[date_col] + pd.Timedelta(days=days)

    # merge_asof with direction="forward": match each anchor to first future date >= anchor
    matched = pd.merge_asof(
        base.sort_values("_anchor"),
        future.rename(columns={date_col: "_future_date", price_col: "_future_price"}).sort_values("_future_date"),
        left_on="_anchor",
        right_on="_future_date",
        direction="forward",
        allow_exact_matches=True,
    )

    # Restore original row order
    matched = matched.sort_values("_row_id")
    out = matched["_future_price"]

    # Ensure output aligns to original df (including any rows dropped for bad dates)
    # If original df had NaT dates, those become NaN targets.
    result = pd.Series(index=np.arange(len(df)), dtype="float64")
    result.loc[matched["_row_id"].values] = out.values

    return result

def _compute_atr_14(df: pd.DataFrame, high="high", low="low", close="front_month_price") -> pd.Series:
    prev_close = df[close].shift(1)
    tr1 = df[high] - df[low]
    tr2 = (df[high] - prev_close).abs()
    tr3 = (df[low] - prev_close).abs()
    tr = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    return pd.Series(tr).rolling(14).mean()


def _data_dictionary_markdown(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Data Dictionary (features_daily.csv)")
    lines.append("")
    lines.append("| Column | Type | Description |")
    lines.append("|---|---|---|")

    descriptions = {
        "commodity": "Commodity label (e.g., coffee).",
        "date": "ISO-8601 date (YYYY-MM-DD).",
        "ticker": "Yahoo Finance ticker used as price proxy (e.g., KC=F).",
        "open": "Daily open price (Yahoo).",
        "high": "Daily high price (Yahoo).",
        "low": "Daily low price (Yahoo).",
        "close": "Daily close price (Yahoo).",
        "adj_close": "Adjusted close (Yahoo, if present).",
        "volume": "Daily volume (Yahoo, if present).",
        "front_month_price": "Canonical daily price series used for ML labeling (mapped from close).",
        "return_1d": "1-day simple return of front_month_price.",
        "log_return_1d": "1-day log return of front_month_price.",
        "vol_7d": "7-day rolling std dev of log returns.",
        "vol_14d": "14-day rolling std dev of log returns.",
        "atr_14": "14-day Average True Range (ATR) computed from high/low/close.",
        "target_price_t_plus_14": "Front_month_price on date+14 calendar days (NULL for last 14 days).",
    }

    for c in df.columns:
        dtype = str(df[c].dtype)
        desc = descriptions.get(c, "Derived/exogenous feature (see pipeline notes).")
        lines.append(f"| {c} | {dtype} | {desc} |")

    lines.append("")
    lines.append("Notes:")
    lines.append("- Yahoo Finance is used as a free proxy data source and may not match official exchange settlement.")
    lines.append("- Missing values are left blank in CSV output (not imputed).")
    return "\n".join(lines)

# ----------------------------
# Weather schema + enforcement
# ----------------------------

WX_COMPACT_SCHEMA_COLUMNS: list[str] = [
    # Base
    "wx_location_count",
    "wx_temperature_2m_max_wavg",
    "wx_temperature_2m_min_wavg",
    "wx_precipitation_sum_wavg",
    "wx_wind_speed_10m_max_wavg",

    # Rollups (means for temp/wind; sums for precip)
    "wx_temperature_2m_max_wavg_mean_7d",
    "wx_temperature_2m_max_wavg_mean_14d",
    "wx_temperature_2m_max_wavg_mean_30d",

    "wx_temperature_2m_min_wavg_mean_7d",
    "wx_temperature_2m_min_wavg_mean_14d",
    "wx_temperature_2m_min_wavg_mean_30d",

    "wx_precipitation_sum_wavg_sum_7d",
    "wx_precipitation_sum_wavg_sum_14d",
    "wx_precipitation_sum_wavg_sum_30d",

    "wx_wind_speed_10m_max_wavg_mean_7d",
    "wx_wind_speed_10m_max_wavg_mean_14d",
    "wx_wind_speed_10m_max_wavg_mean_30d",

    # Lags (base only)
    "wx_location_count_lag_7d",
    "wx_location_count_lag_14d",

    "wx_temperature_2m_max_wavg_lag_7d",
    "wx_temperature_2m_max_wavg_lag_14d",

    "wx_temperature_2m_min_wavg_lag_7d",
    "wx_temperature_2m_min_wavg_lag_14d",

    "wx_precipitation_sum_wavg_lag_7d",
    "wx_precipitation_sum_wavg_lag_14d",

    "wx_wind_speed_10m_max_wavg_lag_7d",
    "wx_wind_speed_10m_max_wavg_lag_14d",
]

def enforce_compact_weather_schema(
    df: pd.DataFrame,
    *,
    keep_weather: bool = True,
    reorder: bool = True,
    wx_schema: Sequence[str] = (),
    extra_keep: Iterable[str] = (),
    target_cols: Sequence[str] = ("target_price_t_plus_14",),
) -> pd.DataFrame:
    """
    Enforce a compact allowlist of wx_* columns and deterministic ordering.

    Ordering (when reorder=True):
      1) all non-wx, non-target columns in their ORIGINAL order
      2) wx columns in wx_schema order (only those present)
      3) any remaining wx_* columns that survived allowlisting (sorted) ("wx_extras")
      4) target_cols in the exact order provided, ABSOLUTELY LAST

    Behavior:
      - If keep_weather=False: drop ALL columns that start with 'wx_'.
      - Else: drop any 'wx_' columns not in wx_schema (+ extra_keep).
      - Defensive: drop accidental “lag of rolling” features by name pattern.

    Notes:
      - Missing wx columns in wx_schema are skipped (no error).
      - Missing targets are skipped (no error).
      - Targets are never interleaved: they are appended at the very end.
    """
    out = df.copy()

    # Normalize targets; preserve provided order; ignore blanks
    target_cols = tuple([c for c in target_cols if c])

    wx_cols = [c for c in out.columns if c.startswith("wx_")]
    if wx_cols and not keep_weather:
        return out.drop(columns=wx_cols)

    if wx_cols and keep_weather:
        allow = set(wx_schema) | set(extra_keep)

        # Drop unlisted wx_ columns
        drop_cols = [c for c in wx_cols if c not in allow]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        # Defensive: remove accidental “lag of rolling” features if upstream naming changes
        forbidden = re.compile(r"_mean_\d+d_lag_\d+d|_sum_\d+d_lag_\d+d")
        bad = [c for c in out.columns if c.startswith("wx_") and forbidden.search(c)]
        if bad:
            out = out.drop(columns=bad)

    if not reorder:
        return out

    # Targets present in df, in the exact order requested
    targets_present = [c for c in target_cols if c in out.columns]

    # Keep everything else (non-targets) in deterministic order:
    # - non-wx columns in original order
    # - wx in schema order
    # - wx extras sorted
    non_targets = [c for c in out.columns if c not in targets_present]

    non_wx = [c for c in non_targets if not c.startswith("wx_")]
    wx_ordered = [c for c in wx_schema if c in non_targets]
    wx_extras = sorted([c for c in non_targets if c.startswith("wx_") and c not in wx_ordered])

    cols = non_wx + wx_ordered + wx_extras + targets_present
    return out.loc[:, cols]

# ----------------------------
# Weather feature engineering
# ----------------------------

def _weighted_weather_daily_aggregate(
    weather_df: pd.DataFrame,
    *,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Input: per-location daily weather with columns:
      date, location, temperature_2m_max, temperature_2m_min, precipitation_sum, windspeed_10m_max, ...
    Output: one row per date with:
      wx_<var>_wavg, wx_location_count
    """
    if weather_df.empty:
        return pd.DataFrame()

    w = weather_df.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce").dt.date.astype(str)
    w = w.dropna(subset=["date", "location"]).reset_index(drop=True)

    # numeric coercion for base vars
    for c in w.columns:
        if c in ("date", "location"):
            continue
        w[c] = pd.to_numeric(w[c], errors="coerce")

    w["weight"] = w["location"].map(weights).fillna(0.0)

    base_vars = [c for c in w.columns if c not in {"date", "location", "weight"}]
    parts = []
    for col in base_vars:
        num = (w[col] * w["weight"]).groupby(w["date"]).sum()
        den = w["weight"].groupby(w["date"]).sum().replace(0.0, np.nan)
        parts.append((num / den).rename(f"wx_{col}_wavg"))

    out = pd.concat(parts, axis=1).reset_index().rename(columns={"index": "date"})
    out["wx_location_count"] = w.groupby("date")["location"].nunique().values
    return out


def _add_weather_rollups(
    df_in: pd.DataFrame,
    *,
    cols: Sequence[str],
    windows: Sequence[int] = (7, 14, 30),
) -> pd.DataFrame:
    """
    Rolling features:
      - precip -> rolling sum
      - temp/wind -> rolling mean
    """
    out = df_in.copy()
    if not cols:
        return out

    roll = {}
    for c in cols:
        if c not in out.columns:
            continue
        for w in windows:
            if "precip" in c or "precipitation" in c:
                roll[f"{c}_sum_{w}d"] = out[c].rolling(w, min_periods=1).sum()
            else:
                roll[f"{c}_mean_{w}d"] = out[c].rolling(w, min_periods=1).mean()

    if roll:
        out = pd.concat([out, pd.DataFrame(roll, index=out.index)], axis=1)

    return out


def _add_weather_lags(
    df_in: pd.DataFrame,
    *,
    cols: Sequence[str],
    lags: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    out = df_in.copy()
    if not cols:
        return out

    lagged = {}
    for c in cols:
        if c not in out.columns:
            continue
        for lag in lags:
            lagged[f"{c}_lag_{lag}d"] = out[c].shift(lag)

    if lagged:
        out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    return out


# ----------------------------
# Exogenous missingness policy
# ----------------------------

def _apply_exogenous_missingness_policy(
    df: pd.DataFrame,
    *,
    exogenous_cols: list[str],
    protected_cols: set[str],
    max_missing_pct: float = 40.0,
    do_backfill: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    - Drops exogenous cols with missing% > threshold (except protected).
    - ffill/bfill remaining exogenous cols (except protected).
    """
    summary = {"dropped_exogenous_cols": [], "exogenous_missing_pct": {}}
    exogenous_cols = [c for c in exogenous_cols if c in df.columns]
    if not exogenous_cols:
        return df, summary

    miss = (df[exogenous_cols].isna().mean() * 100.0).to_dict()
    summary["exogenous_missing_pct"] = {k: float(v) for k, v in miss.items()}

    to_drop = [
        c for c in exogenous_cols
        if c not in protected_cols and miss.get(c, 0.0) > max_missing_pct
    ]
    if to_drop:
        df = df.drop(columns=to_drop)
        summary["dropped_exogenous_cols"] = to_drop

    remaining = [c for c in exogenous_cols if c in df.columns and c not in protected_cols]
    if remaining:
        df[remaining] = df[remaining].ffill()
        if do_backfill:
            df[remaining] = df[remaining].bfill()

    return df, summary

@tool("build_features_daily_tool")
def build_features_daily_tool(
    topic: str,
    raw_ohlcv_path: str = "",
    out_processed_path: str = "data/processed/features_daily.csv",
    days_ahead: int = 14,
) -> str:
    """
    Build ML-ready daily features CSV for a commodity using Yahoo OHLCV as canonical price.
    Upgraded to match your LangGraph output style for weather:
      - production-weighted daily aggregates (wx_*_wavg)
      - rolling features (7/14/30)
      - explicit lags (7/14) on base wx signals
      - compact wx schema enforcement for linear regression
      - deterministic exogenous missingness policy (drop >40%, ffill/bfill)
    Returns: CSV text + markdown data dictionary (CrewAI task contract).
    """
    if not raw_ohlcv_path:
        raw_ohlcv_path = f"data/raw/{topic}_yahoo_ohlcv_1d.csv"

    raw_path = Path(raw_ohlcv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw OHLCV file not found: {raw_ohlcv_path}")

    df = pd.read_csv(raw_path)

    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw OHLCV: {sorted(missing)}")

    # Normalize identity
    df["commodity"] = df.get("commodity", topic)
    df["ticker"] = df.get("ticker", "UNKNOWN")

    # Parse date + sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # Coerce core numerics
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Canonical price
    df["front_month_price"] = pd.to_numeric(df["close"], errors="coerce")

    # Market features (keep your original naming to avoid breaking downstream tasks)
    df["return_1d"] = df["front_month_price"].pct_change()
    df["log_return_1d"] = np.log(df["front_month_price"]).diff()
    df["vol_7d"] = df["log_return_1d"].rolling(7).std()
    df["vol_14d"] = df["log_return_1d"].rolling(14).std()

    # ATR (if high/low exist)
    if {"high", "low"}.issubset(df.columns):
        df["atr_14"] = _compute_atr_14(df, high="high", low="low", close="front_month_price")
    else:
        df["atr_14"] = np.nan

    # Convert date to ISO string after feature calcs
    df["date"] = df["date"].dt.date.astype(str)

    # ----------------------------
    # Optional exogenous merges
    # ----------------------------
    merged_exogenous_cols: list[str] = []

    exogenous_candidates = {
        "macro_fx": f"data/processed/{topic}_macro_fx_daily.csv",
        "weather": f"data/processed/{topic}_weather_daily.csv",
        "cot": f"data/processed/{topic}_cot_daily.csv",
    }

    # Weather weights (override by setting env or editing here)
    # Match your 6-region “coffee” defaults from the LangGraph side.
    default_weather_weights = {
        "Brazil_MinasGerais": 0.35,
        "Brazil_EspiritoSanto": 0.15,
        "Vietnam_CentralHighlands": 0.20,
        "Colombia_Huila": 0.15,
        "Indonesia_Lampung": 0.10,
        "Ethiopia_Oromia": 0.05,
        # Also tolerate alternate naming from older pipelines:
        "MinasGerais": 0.35,
        "EspiritoSanto": 0.15,
        "CentralHighlands": 0.20,
        "Huila": 0.15,
        "Lampung": 0.10,
        "Oromia": 0.05,
    }

    for name, path in exogenous_candidates.items():
        ex = _read_csv_if_exists(path)
        if ex is None or ex.empty:
            continue
        if "date" not in ex.columns:
            continue

        ex["date"] = pd.to_datetime(ex["date"], errors="coerce").dt.date.astype(str)

        if name == "weather":
            # If weather is per-location (Open-Meteo raw style), aggregate first.
            if "location" in ex.columns:
                wx_daily = _weighted_weather_daily_aggregate(ex, weights=default_weather_weights)

                # Merge aggregated wx_* into df
                df = df.merge(wx_daily, on="date", how="left")
                newly_added = [c for c in wx_daily.columns if c != "date"]
                merged_exogenous_cols.extend(newly_added)

                # Rolling + lags on base signals only
                base_for_roll = [
                    "wx_temperature_2m_max_wavg",
                    "wx_temperature_2m_min_wavg",
                    "wx_precipitation_sum_wavg",
                    "wx_wind_speed_10m_max_wavg",
                ]
                df = _add_weather_rollups(df, cols=base_for_roll, windows=(7, 14, 30))

                base_for_lag = base_for_roll + ["wx_location_count"]
                df = _add_weather_lags(df, cols=base_for_lag, lags=(7, 14))

                merged_exogenous_cols.extend([c for c in df.columns if c.startswith("wx_") and c not in merged_exogenous_cols])
            else:
                # Already aggregated daily weather; merge and then enforce roll/lag/schema downstream
                merge_cols = [c for c in ex.columns if c != "date"]
                df = df.merge(ex[["date"] + merge_cols], on="date", how="left")
                merged_exogenous_cols.extend([c for c in merge_cols if c.startswith("wx_")])

        else:
            # macro_fx / cot: merge on date; avoid collisions
            merge_cols = [c for c in ex.columns if c != "date"]
            df = df.merge(ex[["date"] + merge_cols], on="date", how="left")
            merged_exogenous_cols.extend(merge_cols)

    # ----------------------------
    # Targets (calendar-day +7 and +14)
    # ----------------------------
    df["target_price_t_plus_7"]  = target_next_trading_day_on_or_after(df, days=7)
    df["target_price_t_plus_14"] = target_next_trading_day_on_or_after(df, days=14)

    dt = pd.to_datetime(df["date"], errors="coerce")
    max_dt = dt.max()
    df.loc[dt > (max_dt - pd.Timedelta(days=7)),  "target_price_t_plus_7"]  = np.nan
    df.loc[dt > (max_dt - pd.Timedelta(days=14)), "target_price_t_plus_14"] = np.nan

    # ----------------------------
    # Deterministic exogenous missingness policy
    # ----------------------------
    protected_cols = {
        "commodity", "ticker", "date",
        "front_month_price", "return_1d", "log_return_1d", "vol_7d", "vol_14d", "atr_14",
        "open", "high", "low", "close", "adj_close", "volume", "target_price_t_plus_7",
        "target_price_t_plus_14",
    }
    df, policy_summary = _apply_exogenous_missingness_policy(
        df,
        exogenous_cols=list(dict.fromkeys(merged_exogenous_cols)),
        protected_cols=protected_cols,
        max_missing_pct=40.0,
        do_backfill=True,
    )

    # ----------------------------
    # Enforce compact wx schema (linear regression)
    # ----------------------------
    df = enforce_compact_weather_schema(
        df,
        keep_weather=True,
        reorder=True,
        wx_schema=WX_COMPACT_SCHEMA_COLUMNS,
        target_cols=("target_price_t_plus_7", "target_price_t_plus_14"),
        target_last=True,
    )

    # ----------------------------
    # Quality checks: duplicates + numeric coercion
    # ----------------------------
    if df.duplicated(subset=["commodity", "date"]).any():
        raise ValueError("Duplicate rows detected for (commodity, date).")

    key_cols = {"commodity", "date", "ticker"}
    for c in df.columns:
        if c in key_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------------------
    # Write processed CSV (blank for missing)
    # ----------------------------
    out_path = Path(out_processed_path)
    _ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, na_rep="")

    csv_text = out_path.read_text(encoding="utf-8")
    dd_md = _data_dictionary_markdown(df)

    return f"{csv_text}\n\n{dd_md}"
