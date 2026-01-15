import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred
from crewai.tools import BaseTool

load_dotenv(override=True)


class FredMacroFXTool(BaseTool):
    name: str = "fred_macro_fx_tool"
    description: str = (
        "Fetches daily macro + FX series from FRED and supplemental World Bank FX "
        "for the requested date range. Writes a CSV to disk and returns the file path."
    )

    def _run(
        self,
        topic: str = "coffee",
        start_date: str = "2015-01-01",
        end_date: str = "2025-01-01",
        out_path: Optional[str] = None,
    ) -> str:
        fred_key = os.getenv("FRED_API_KEY")
        if not fred_key:
            raise ValueError("Missing FRED_API_KEY in environment")

        fred = Fred(api_key=fred_key)

        df = self.get_daily_macro(
            fred=fred,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            raise ValueError("No macro/FX data returned from FRED/World Bank")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df["date"] = df["date"].dt.date.astype(str)

        # Write output
        if out_path is None:
            out_path = f"data/processed/{topic}_macro_fx_daily.csv"

        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False, na_rep="")

        return str(out_file)

    def get_worldbank_exchange_rate(
        self,
        country_code: str,
        currency_name: str,
        *,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Annual WB FX forward-filled to daily."""
        try:
            url = (
                "https://api.worldbank.org/v2/country/"
                f"{country_code}/indicator/PA.NUS.FCRF"
                f"?format=json&per_page=500&date={start_year}:{end_year}"
            )
            response = requests.get(url, timeout=15)
            data = response.json()

            if len(data) > 1 and data[1]:
                records = []
                for item in data[1]:
                    val = item.get("value")
                    if val is None:
                        continue
                    year = int(item["date"])
                    # Mid-year anchor for annual data
                    dt = pd.Timestamp(f"{year}-06-30")
                    records.append({"date": dt, f"{currency_name}_usd": val})

                if not records:
                    return pd.DataFrame()

                df = pd.DataFrame(records)
                df = df.set_index("date").sort_index().resample("D").ffill().reset_index()
                return df

        except Exception:
            return pd.DataFrame()

        return pd.DataFrame()

    def get_daily_macro(self, fred: Fred, start_date: str, end_date: str) -> pd.DataFrame:
        """Macro + FX daily series, merged on date."""
        fred_series = {
            "DEXBZUS": "brl_usd",
            "DEXINUS": "inr_usd",
            "DEXCHUS": "cny_usd",
            "DEXTHUS": "thb_usd",
            "DEXMXUS": "mxn_usd",
            "DEXUSAL": "aud_usd",
            "DEXUSEU": "eur_usd",
            "DEXSFUS": "zar_usd",
            "DEXJPUS": "jpy_usd",
            "DEXSZUS": "chf_usd",
            "DEXKOUS": "krw_usd",
            "DEXUSUK": "gbp_usd",
            "DCOILWTICO": "oil_wti",
            "DTWEXBGS": "usd_index",
            "DGS10": "us_10yr_rate",
        }

        dfs: List[pd.DataFrame] = []

        # FRED daily series
        for code, colname in fred_series.items():
            try:
                s = fred.get_series(code, observation_start=start_date, observation_end=end_date)
                df = pd.DataFrame({"date": s.index, colname: s.values})
                dfs.append(df)
                time.sleep(0.2)
            except Exception:
                # Keep going; missing series should not hard-fail the whole pipeline
                continue

        # World Bank annual FX (forward-filled)
        start_year = int(pd.Timestamp(start_date).year)
        end_year = int(pd.Timestamp(end_date).year)

        worldbank_currencies: List[Tuple[str, str]] = [
            ("VNM", "vnd"),
            ("COL", "cop"),
            ("IDN", "idr"),
            ("ETH", "etb"),
            ("HND", "hnl"),
            ("UGA", "ugx"),
            ("PER", "pen"),
            ("CAF", "xaf"),
            ("GTM", "gtq"),
            ("GIN", "gnf"),
            ("NIC", "nio"),
            ("CRI", "crc"),
            ("TZA", "tzs"),
            ("KEN", "kes"),
            ("LAO", "lak"),
            ("PAK", "pkr"),
            ("PHL", "php"),
            ("EGY", "egp"),
            ("ARG", "ars"),
            ("RUS", "rub"),
            ("TUR", "try"),
            ("UKR", "uah"),
            ("IRN", "irr"),
            ("BLR", "byn"),
        ]

        for cc, cur in worldbank_currencies:
            wb = self.get_worldbank_exchange_rate(
                cc,
                cur,
                start_year=start_year,
                end_year=end_year,
            )
            if not wb.empty:
                dfs.append(wb)

        if not dfs:
            return pd.DataFrame()

        # Outer merge all on date
        out = dfs[0]
        for d in dfs[1:]:
            out = out.merge(d, on="date", how="outer")

        # Filter to requested date range
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))]
        return out


fred_macro_fx_tool = FredMacroFXTool()