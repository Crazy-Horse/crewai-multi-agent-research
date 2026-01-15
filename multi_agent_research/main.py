#!/usr/bin/env python
import warnings
from datetime import datetime

from multi_agent_research.crew import MultiAgentResearch

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

DEFAULT_WEATHER_LOCATIONS = [
    {"name": "Brazil_MinasGerais", "latitude": -18.5, "longitude": -44.6},
    {"name": "Brazil_EspiritoSanto", "latitude": -19.5, "longitude": -40.6},
    {"name": "Vietnam_CentralHighlands", "latitude": 12.7, "longitude": 108.1},
    {"name": "Colombia_Huila", "latitude": 2.5, "longitude": -75.6},
    {"name": "Indonesia_Lampung", "latitude": -5.4, "longitude": 105.3},
    {"name": "Ethiopia_Oromia", "latitude": 8.5, "longitude": 39.0},
]

def run():
    inputs = {
        "topic": "coffee", 
        "current_year": datetime.now().year,

        "start_date": "2015-01-01",
        "end_date": "2025-01-01",

        "ticker": "KC=F",

        "weather_locations": DEFAULT_WEATHER_LOCATIONS,
    }

    try:
        MultiAgentResearch().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e

if __name__ == "__main__":
    run()
