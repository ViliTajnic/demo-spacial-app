from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from detection import haversine_m
except ModuleNotFoundError:
    from app.detection import haversine_m


PATTERN_LIBRARY = [
    {
        "pattern_id": 1,
        "description": "Safe Traffic Detour",
        "decision_hint": "DEPRIORITIZE",
        "speed_score": 0.75,
        "dwell_score": 0.017,
        "proximity_score": 0.10,
    },
    {
        "pattern_id": 2,
        "description": "High Risk Loitering",
        "decision_hint": "ELEVATE",
        "speed_score": 0.016,
        "dwell_score": 0.50,
        "proximity_score": 0.80,
    },
]




def stable_zone_id(zone_name: str) -> int:
    total = 0
    for idx, char in enumerate(zone_name, start=1):
        total += idx * ord(char)
    return total


def attach_zone_ids(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return rules.copy()
    output = rules.copy()
    output["zone_id"] = output["zone_name"].apply(stable_zone_id)
    return output


PRESENTATION_ZONE = {
    "participant_id": "SCENARIO_RULE",
    "rule_type": "EXCLUSION_ZONE",
    "zone_name": "Central High School",
    "zone_lat": 52.370216,
    "zone_lon": 4.895168,
    "radius_m": 120.0,
}


def get_pattern_library() -> pd.DataFrame:
    df = pd.DataFrame(PATTERN_LIBRARY)
    df["behavior_vector"] = df.apply(
        lambda row: vector_to_literal([row["speed_score"], row["dwell_score"], row["proximity_score"]]),
        axis=1,
    )
    return df


def vector_to_literal(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _meters_to_lat(meters: float) -> float:
    return meters / 111_320


def _meters_to_lon(meters: float, lat: float) -> float:
    return meters / (111_320 * max(math.cos(math.radians(lat)), 0.2))


def _make_track(
    participant_id: str,
    city: str,
    start_ts: datetime,
    points: list[tuple[float, float, float, int]],
) -> list[dict]:
    rows = []
    for offset_min, lat, lon, speed_kmh in points:
        rows.append(
            {
                "participant_id": participant_id,
                "city": city,
                "event_ts": start_ts + timedelta(minutes=offset_min),
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "speed_kmh": round(float(speed_kmh), 2),
            }
        )
    return rows


def generate_presentation_scenarios(background_participants: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_lat = PRESENTATION_ZONE["zone_lat"]
    zone_lon = PRESENTATION_ZONE["zone_lon"]
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=35)

    route_offsets = np.linspace(-220, 220, 12)
    routine_points = []
    for idx, east_m in enumerate(route_offsets):
        lat = zone_lat + _meters_to_lat(12 * math.sin(idx / 3))
        lon = zone_lon + _meters_to_lon(float(east_m), zone_lat)
        routine_points.append((idx, lat, lon, 72 - (idx % 3) * 4))

    loiter_points = []
    for idx in range(26):
        north_m = 14 * math.sin(idx / 2.5)
        east_m = 10 * math.cos(idx / 2.7)
        lat = zone_lat + _meters_to_lat(north_m)
        lon = zone_lon + _meters_to_lon(east_m, zone_lat)
        loiter_points.append((idx, lat, lon, 1.2 + (idx % 4) * 0.3))

    outside_points = []
    for idx, east_m in enumerate(np.linspace(600, 1500, 10)):
        north_m = 820 + idx * 10
        lat = zone_lat + _meters_to_lat(float(north_m))
        lon = zone_lon + _meters_to_lon(float(east_m), zone_lat)
        outside_points.append((idx * 2, lat, lon, 58 + (idx % 2) * 4))

    scenario_locations = pd.DataFrame(
        _make_track("S0001", "Amsterdam", start_ts, routine_points)
        + _make_track("S0002", "Amsterdam", start_ts, loiter_points)
        + _make_track("S0003", "Amsterdam", start_ts, outside_points)
    )

    if background_participants > 0:
        try:
            from simulator import SimulationConfig, generate_locations, generate_rules
        except ModuleNotFoundError:
            from app.simulator import SimulationConfig, generate_locations, generate_rules

        background_cfg = SimulationConfig(
            participants=background_participants,
            duration_hours=4,
            point_interval_min=5,
            random_seed=99,
        )
        background_locations = generate_locations(background_cfg)
        background_rules = generate_rules(sorted(background_locations["participant_id"].unique()))
    else:
        background_locations = pd.DataFrame(columns=scenario_locations.columns)
        background_rules = pd.DataFrame(columns=["participant_id", "rule_type", "zone_name", "zone_lat", "zone_lon", "radius_m"])

    locations = pd.concat([scenario_locations, background_locations], ignore_index=True).sort_values(["participant_id", "event_ts"])

    scenario_rules = pd.DataFrame(
        [
            {
                **PRESENTATION_ZONE,
                "participant_id": participant_id,
            }
            for participant_id in sorted(scenario_locations["participant_id"].unique())
        ]
    )
    rules = pd.concat([scenario_rules, background_rules], ignore_index=True)
    rules = attach_zone_ids(rules)
    return locations.reset_index(drop=True), rules.reset_index(drop=True)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 - similarity


def build_behavior_vectors(locations: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if locations.empty or rules.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "zone_name",
                "zone_id",
                "speed_score",
                "dwell_score",
                "proximity_score",
                "inside_zone",
                "behavior_vector",
            ]
        )

    rules = attach_zone_ids(rules)
    merged = locations.merge(rules, on="participant_id", how="left")
    merged["distance_m"] = merged.apply(
        lambda row: haversine_m(row["lat"], row["lon"], row["zone_lat"], row["zone_lon"]),
        axis=1,
    )
    merged["inside_zone"] = merged["distance_m"] <= merged["radius_m"]
    merged = merged.sort_values(["participant_id", "event_ts"])

    rows = []
    for participant_id, frame in merged.groupby("participant_id"):
        frame = frame.reset_index(drop=True)
        zone_name = frame.iloc[0]["zone_name"]
        radius_m = float(frame.iloc[0]["radius_m"])
        speed_score = float(np.clip(frame["speed_kmh"].median() / 80.0, 0.0, 1.0))

        in_zone = frame[frame["inside_zone"]].copy()
        if in_zone.empty:
            dwell_minutes = 0.0
            proximity_score = 0.0
            inside_zone = 0
        else:
            deltas = in_zone["event_ts"].diff().dt.total_seconds().div(60).fillna(1)
            dwell_minutes = float(deltas.clip(lower=1, upper=10).sum())
            closeness = (1 - (in_zone["distance_m"] / max(radius_m, 1.0))).clip(lower=0, upper=1)
            stationary_weight = 1 - (in_zone["speed_kmh"] / 80.0).clip(lower=0, upper=1)
            proximity_score = float(np.clip((closeness * stationary_weight).mean(), 0.0, 1.0))
            inside_zone = 1

        dwell_score = float(np.clip(dwell_minutes / 50.0, 0.0, 1.0))
        vector_values = [speed_score, dwell_score, proximity_score]
        rows.append(
            {
                "participant_id": participant_id,
                "zone_id": int(frame.iloc[0]["zone_id"]),
                "zone_name": zone_name,
                "speed_score": round(speed_score, 6),
                "dwell_score": round(dwell_score, 6),
                "proximity_score": round(proximity_score, 6),
                "inside_zone": inside_zone,
                "behavior_vector": vector_to_literal(vector_values),
            }
        )

    return pd.DataFrame(rows)


def apply_behavior_decisions(behavior_vectors: pd.DataFrame) -> pd.DataFrame:
    patterns = get_pattern_library()
    if behavior_vectors.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "zone_name",
                "matched_pattern",
                "similarity_score",
                "decision",
                "speed_score",
                "dwell_score",
                "proximity_score",
            ]
        )

    rows = []
    for _, subject in behavior_vectors.iterrows():
        subject_vector = np.array(
            [subject["speed_score"], subject["dwell_score"], subject["proximity_score"]],
            dtype=float,
        )
        scored_patterns = []
        for _, pattern in patterns.iterrows():
            pattern_vector = np.array(
                [pattern["speed_score"], pattern["dwell_score"], pattern["proximity_score"]],
                dtype=float,
            )
            scored_patterns.append(
                {
                    "description": pattern["description"],
                    "decision_hint": pattern["decision_hint"],
                    "distance": _cosine_distance(subject_vector, pattern_vector),
                }
            )

        best = sorted(scored_patterns, key=lambda item: item["distance"])[0]
        if not int(subject["inside_zone"]):
            decision = "NO_ACTION"
            matched_pattern = "N/A"
            similarity_score = None
        elif best["decision_hint"] == "DEPRIORITIZE" and best["distance"] <= 0.08:
            decision = "DEPRIORITIZE"
            matched_pattern = best["description"]
            similarity_score = round(best["distance"], 4)
        elif best["decision_hint"] == "ELEVATE" and best["distance"] <= 0.2:
            decision = "ELEVATE"
            matched_pattern = best["description"]
            similarity_score = round(best["distance"], 4)
        else:
            decision = "REVIEW"
            matched_pattern = best["description"]
            similarity_score = round(best["distance"], 4)

        rows.append(
            {
                "participant_id": subject["participant_id"],
                "zone_name": subject["zone_name"],
                "matched_pattern": matched_pattern,
                "similarity_score": similarity_score,
                "decision": decision,
                "speed_score": subject["speed_score"],
                "dwell_score": subject["dwell_score"],
                "proximity_score": subject["proximity_score"],
                "inside_zone": subject["inside_zone"],
            }
        )

    return pd.DataFrame(rows)
