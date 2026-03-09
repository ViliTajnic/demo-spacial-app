from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random
from typing import List

import numpy as np
import pandas as pd


EU_CITY_SEEDS = {
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.52, 13.405),
    "Madrid": (40.4168, -3.7038),
    "Rome": (41.9028, 12.4964),
    "Amsterdam": (52.3676, 4.9041),
    "Vienna": (48.2082, 16.3738),
    "Prague": (50.0755, 14.4378),
    "Warsaw": (52.2297, 21.0122),
    "Lisbon": (38.7223, -9.1393),
    "Brussels": (50.8503, 4.3517),
}


@dataclass
class SimulationConfig:
    participants: int = 30
    duration_hours: int = 6
    point_interval_min: int = 5
    random_seed: int = 42


def _jitter(lat: float, lon: float, step_scale: float = 0.0018) -> tuple[float, float]:
    return lat + np.random.normal(0, step_scale), lon + np.random.normal(0, step_scale)


def _meters_to_degree_offsets(north_m: float, east_m: float, base_lat: float) -> tuple[float, float]:
    lat_offset = north_m / 111_320
    lon_offset = east_m / (111_320 * max(math.cos(math.radians(base_lat)), 0.2))
    return lat_offset, lon_offset


def _bearing_rotate(north_m: float, east_m: float, degrees: float) -> tuple[float, float]:
    angle = math.radians(degrees)
    rotated_north = north_m * math.cos(angle) - east_m * math.sin(angle)
    rotated_east = north_m * math.sin(angle) + east_m * math.cos(angle)
    return rotated_north, rotated_east


def _build_template_offsets(template_df: pd.DataFrame) -> pd.DataFrame:
    template = template_df.sort_values("event_ts").reset_index(drop=True).copy()
    template["event_ts"] = pd.to_datetime(template["event_ts"], utc=True)

    first_lat = float(template.iloc[0]["lat"])
    first_lon = float(template.iloc[0]["lon"])
    first_ts = template.iloc[0]["event_ts"]

    north_m = (template["lat"].astype(float) - first_lat) * 111_320
    east_m = (template["lon"].astype(float) - first_lon) * 111_320 * np.cos(np.radians(first_lat))
    minutes_from_start = (template["event_ts"] - first_ts).dt.total_seconds() / 60.0

    out = pd.DataFrame(
        {
            "minutes_from_start": minutes_from_start,
            "north_m": north_m,
            "east_m": east_m,
        }
    )
    return out


def generate_locations(cfg: SimulationConfig) -> pd.DataFrame:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    city_names = list(EU_CITY_SEEDS.keys())
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=cfg.duration_hours)
    timestamps: List[datetime] = [
        start_ts + timedelta(minutes=i * cfg.point_interval_min)
        for i in range((cfg.duration_hours * 60) // cfg.point_interval_min)
    ]

    records = []
    for p in range(1, cfg.participants + 1):
        city = city_names[(p - 1) % len(city_names)]
        base_lat, base_lon = EU_CITY_SEEDS[city]

        lat, lon = base_lat, base_lon
        for ts in timestamps:
            lat, lon = _jitter(lat, lon)
            speed_kmh = max(0.5, np.random.normal(3.5, 1.2))
            records.append(
                {
                    "participant_id": f"P{p:04d}",
                    "city": city,
                    "event_ts": ts,
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "speed_kmh": round(float(speed_kmh), 2),
                }
            )

    return pd.DataFrame(records)


def generate_locations_from_template(template_df: pd.DataFrame, participants: int, random_seed: int = 42) -> pd.DataFrame:
    if template_df.empty:
        return pd.DataFrame(columns=["participant_id", "city", "event_ts", "lat", "lon", "speed_kmh"])

    random.seed(random_seed)
    np.random.seed(random_seed)

    template = _build_template_offsets(template_df)
    city_names = list(EU_CITY_SEEDS.keys())
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=float(template["minutes_from_start"].max())
    )

    records = []
    for participant_num in range(1, participants + 1):
        city = city_names[(participant_num - 1) % len(city_names)]
        base_lat, base_lon = EU_CITY_SEEDS[city]
        anchor_north = random.uniform(-1200, 1200)
        anchor_east = random.uniform(-1200, 1200)
        rotation = random.uniform(-35, 35)
        scale = random.uniform(0.85, 1.2)

        for idx, row in template.iterrows():
            north_m, east_m = _bearing_rotate(
                float(row["north_m"]) * scale,
                float(row["east_m"]) * scale,
                rotation,
            )
            north_m += anchor_north + np.random.normal(0, 25)
            east_m += anchor_east + np.random.normal(0, 25)
            lat_offset, lon_offset = _meters_to_degree_offsets(north_m, east_m, base_lat)

            speed_kmh = 2.8 if idx == 0 else max(
                0.6,
                np.random.normal(4.2, 1.0),
            )
            records.append(
                {
                    "participant_id": f"T{participant_num:04d}",
                    "city": city,
                    "event_ts": start_ts + timedelta(minutes=float(row["minutes_from_start"])),
                    "lat": round(float(base_lat + lat_offset), 6),
                    "lon": round(float(base_lon + lon_offset), 6),
                    "speed_kmh": round(float(speed_kmh), 2),
                }
            )

    return pd.DataFrame(records)


def generate_device_events(locations: pd.DataFrame) -> pd.DataFrame:
    participants = locations["participant_id"].unique().tolist()
    rows = []
    for p in participants:
        participant_locs = locations[locations["participant_id"] == p]
        battery = random.randint(35, 95)
        for _, row in participant_locs.iterrows():
            battery_drop = random.choice([0, 0, 1, 1, 2])
            battery = max(5, battery - battery_drop)

            tamper = 1 if random.random() < 0.01 else 0
            no_signal = 1 if random.random() < 0.02 else 0

            rows.append(
                {
                    "participant_id": p,
                    "event_ts": row["event_ts"],
                    "battery_pct": battery,
                    "tamper_flag": tamper,
                    "no_signal_flag": no_signal,
                }
            )

    return pd.DataFrame(rows)


def generate_rules(participants: List[str]) -> pd.DataFrame:
    rules = []
    school_zones = [
        {"name": "Paris School Zone", "lat": 48.8602, "lon": 2.3376, "radius_m": 250},
        {"name": "Berlin School Zone", "lat": 52.5179, "lon": 13.3889, "radius_m": 250},
        {"name": "Madrid School Zone", "lat": 40.4172, "lon": -3.706, "radius_m": 250},
    ]
    for p in participants:
        assigned = school_zones[hash(p) % len(school_zones)]
        rules.append(
            {
                "participant_id": p,
                "rule_type": "EXCLUSION_ZONE",
                "zone_name": assigned["name"],
                "zone_lat": assigned["lat"],
                "zone_lon": assigned["lon"],
                "radius_m": assigned["radius_m"],
            }
        )
    return pd.DataFrame(rules)
