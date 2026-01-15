import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError
from crewai.tools import tool

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class Location(BaseModel):
    name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class OpenMeteoRequest(BaseModel):
    locations: List[Location] = Field(..., min_length=1)
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    daily_vars: List[str] = Field(
        default_factory=lambda: [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",  # Open-Meteo field name
        ]
    )
    timezone: str = Field(default="UTC")
    use_archive: Optional[bool] = Field(default=None)


def _choose_endpoint(end_date: date, use_archive: Optional[bool]) -> str:
    if use_archive is True:
        return ARCHIVE_URL
    if use_archive is False:
        return FORECAST_URL
    today = date.today()
    if end_date <= (today - timedelta(days=2)):
        return ARCHIVE_URL
    return FORECAST_URL


def _call(url: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo failed: {r.status_code} {r.reason}. "
            f"Params={json.dumps(params)}. Body={r.text[:500]}"
        )
    return r.json()


def _payload_to_df(location_name: str, payload: Dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        reason = payload.get("reason") or payload.get("message") or "No daily data returned."
        raise RuntimeError(f"Open-Meteo returned no daily data for {location_name}. Reason: {reason}")
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df.insert(1, "location", location_name)
    return df


_DEFAULT_6_COFFEE_REGIONS = [
    {"name": "Brazil_MinasGerais", "latitude": -18.5, "longitude": -44.6},
    {"name": "Brazil_EspiritoSanto", "latitude": -19.5, "longitude": -40.6},
    {"name": "Vietnam_CentralHighlands", "latitude": 12.7, "longitude": 108.1},
    {"name": "Colombia_Huila", "latitude": 2.5, "longitude": -75.6},
    {"name": "Indonesia_Lampung", "latitude": -5.4, "longitude": 105.3},
    {"name": "Ethiopia_Oromia", "latitude": 8.5, "longitude": 39.0},
]


@tool("open_meteo_daily_weather_tool")
def open_meteo_daily_weather_tool(
    topic: str = "coffee",
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    locations: Optional[List[dict]] = None,
    daily_vars: Optional[List[str]] = None,
    timezone: str = "UTC",
    use_archive: Optional[bool] = True,
    out_path: Optional[str] = None,
) -> str:
    """
    Fetch daily weather from Open-Meteo for one or more locations.
    Writes CSV to disk and returns the file path.

    If locations is None/empty, defaults to 6 coffee-growing regions.
    """
    locs = locations or _DEFAULT_6_COFFEE_REGIONS

    try:
        req = OpenMeteoRequest(
            locations=locs,
            start_date=start_date,
            end_date=end_date,
            daily_vars=daily_vars or OpenMeteoRequest.model_fields["daily_vars"].default_factory(),  # type: ignore
            timezone=timezone,
            use_archive=use_archive,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid Open-Meteo inputs: {e}") from e

    s = date.fromisoformat(req.start_date)
    e = date.fromisoformat(req.end_date)
    if e < s:
        raise ValueError("end_date must be >= start_date")

    endpoint = _choose_endpoint(e, req.use_archive)

    frames: List[pd.DataFrame] = []
    for loc in req.locations:
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "daily": ",".join(req.daily_vars),
            "timezone": req.timezone,
        }
        payload = _call(endpoint, params)
        frames.append(_payload_to_df(loc.name, payload))

    out = pd.concat(frames, ignore_index=True)

    # Normalize types
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    out = out.sort_values(["location", "date"])

    for c in out.columns:
        if c in ("date", "location"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # IMPORTANT: normalize Open-Meteo var naming to match your downstream feature naming
    # Your LangGraph pipeline used wind_speed_10m_max downstream.
    if "windspeed_10m_max" in out.columns and "wind_speed_10m_max" not in out.columns:
        out = out.rename(columns={"windspeed_10m_max": "wind_speed_10m_max"})

    # Write file
    if out_path is None:
        out_path = f"data/raw/{topic}_openmeteo_daily.csv"

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False, na_rep="")

    return str(out_file)
