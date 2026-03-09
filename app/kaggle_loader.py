from __future__ import annotations

from io import StringIO
from pathlib import Path

import gpxpy
import pandas as pd


COLUMNS = ["participant_id", "city", "event_ts", "lat", "lon", "speed_kmh"]


def parse_gpx_text(gpx_text: str, participant_id: str, city: str = "EU") -> pd.DataFrame:
    gpx = gpxpy.parse(StringIO(gpx_text))
    rows = []

    for waypoint in gpx.waypoints:
        if waypoint.time is None:
            continue
        rows.append(
            {
                "participant_id": participant_id,
                "city": city,
                "event_ts": waypoint.time,
                "lat": round(float(waypoint.latitude), 6),
                "lon": round(float(waypoint.longitude), 6),
                "speed_kmh": 3.0,
            }
        )

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    continue
                rows.append(
                    {
                        "participant_id": participant_id,
                        "city": city,
                        "event_ts": point.time,
                        "lat": round(float(point.latitude), 6),
                        "lon": round(float(point.longitude), 6),
                        "speed_kmh": 3.0,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows).drop_duplicates(subset=["event_ts", "lat", "lon"])
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    return df.sort_values("event_ts").reset_index(drop=True)


def participant_id_from_filename(filename: str) -> str:
    base = Path(filename).stem.upper().replace(" ", "_")
    return f"K_{base[:18]}"
