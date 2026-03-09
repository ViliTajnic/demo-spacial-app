from __future__ import annotations

import math
from typing import Tuple

import pandas as pd


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def detect_zone_violations(locations: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    merged = locations.merge(rules, on="participant_id", how="inner")
    merged["distance_m"] = merged.apply(
        lambda x: haversine_m(x["lat"], x["lon"], x["zone_lat"], x["zone_lon"]), axis=1
    )
    hit = merged[merged["distance_m"] <= merged["radius_m"]].copy()
    if hit.empty:
        return pd.DataFrame(columns=["participant_id", "event_ts", "alert_type", "severity", "details"])
    hit["alert_type"] = "EXCLUSION_ZONE"
    hit["severity"] = "HIGH"
    hit["details"] = (
        "Inside exclusion zone " + hit["zone_name"] + " distance=" + hit["distance_m"].round(1).astype(str) + "m"
    )
    return hit[["participant_id", "event_ts", "alert_type", "severity", "details"]]


def detect_device_alerts(device_events: pd.DataFrame) -> pd.DataFrame:
    low_battery = device_events[device_events["battery_pct"] < 20].copy()
    low_battery["alert_type"] = "LOW_BATTERY"
    low_battery["severity"] = "MEDIUM"
    low_battery["details"] = "Battery under 20%"

    tamper = device_events[device_events["tamper_flag"] == 1].copy()
    tamper["alert_type"] = "TAMPER"
    tamper["severity"] = "HIGH"
    tamper["details"] = "Tamper signal detected"

    no_signal = device_events[device_events["no_signal_flag"] == 1].copy()
    no_signal["alert_type"] = "NO_SIGNAL"
    no_signal["severity"] = "MEDIUM"
    no_signal["details"] = "No signal event detected"

    out = pd.concat([low_battery, tamper, no_signal], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["participant_id", "event_ts", "alert_type", "severity", "details"])
    return out[["participant_id", "event_ts", "alert_type", "severity", "details"]]


def detect_simultaneous_colocation(
    locations: pd.DataFrame, distance_threshold_m: float = 120.0
) -> pd.DataFrame:
    rows = []
    for ts, frame in locations.groupby("event_ts"):
        data = frame[["participant_id", "lat", "lon"]].to_dict("records")
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                d = haversine_m(data[i]["lat"], data[i]["lon"], data[j]["lat"], data[j]["lon"])
                if d <= distance_threshold_m:
                    rows.append(
                        {
                            "participant_a": data[i]["participant_id"],
                            "participant_b": data[j]["participant_id"],
                            "event_ts": ts,
                            "distance_m": round(d, 2),
                        }
                    )
    return pd.DataFrame(rows)


def score_risk(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame(columns=["participant_id", "risk_score", "risk_band"])

    points = {"HIGH": 35, "MEDIUM": 15, "LOW": 5}
    agg = alerts.copy()
    agg["pts"] = agg["severity"].map(points).fillna(5)

    risk = agg.groupby("participant_id", as_index=False)["pts"].sum()
    risk.rename(columns={"pts": "risk_score"}, inplace=True)

    def band(score: float) -> str:
        if score >= 80:
            return "HIGH"
        if score >= 35:
            return "MEDIUM"
        return "LOW"

    risk["risk_band"] = risk["risk_score"].apply(band)
    return risk


def build_alerts(locations: pd.DataFrame, device_events: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zone = detect_zone_violations(locations, rules)
    dev = detect_device_alerts(device_events)
    frames = [df for df in [zone, dev] if not df.empty]
    if not frames:
        alerts = pd.DataFrame(columns=["participant_id", "event_ts", "alert_type", "severity", "details"])
    else:
        alerts = pd.concat(frames, ignore_index=True).sort_values("event_ts")
    risk = score_risk(alerts)
    return alerts, risk
