from __future__ import annotations

from datetime import datetime, timezone
import json
import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from behavior import (
    apply_behavior_decisions,
    build_behavior_vectors,
    generate_presentation_scenarios,
    get_pattern_library,
)
from detection import build_alerts, detect_simultaneous_colocation
from kaggle_loader import parse_gpx_text, participant_id_from_filename
from llm_client import explain_alert, triage_incidents
from oracle_repo import OracleRepository
from simulator import (
    SimulationConfig,
    generate_device_events,
    generate_locations,
    generate_locations_from_template,
    generate_rules,
)


st.set_page_config(page_title="Sentinel EU Demo", layout="wide")
st.title("Sentinel EU Monitoring Demo")
st.caption("Oracle DB + Local/OCI GenAI demo with EU-focused location data")


WORKSPACES = [
    "Operations Center",
    "Subject Investigation",
    "Hybrid Intelligence",
    "Data & Scenarios",
    "Oracle Lab",
]


def build_dataset(locations: pd.DataFrame, rules: pd.DataFrame) -> dict:
    device_events = generate_device_events(locations)
    alerts, risk = build_alerts(locations, device_events, rules)
    coloc = detect_simultaneous_colocation(locations, distance_threshold_m=120)
    behavior_vectors = build_behavior_vectors(locations, rules)
    hybrid_decisions = apply_behavior_decisions(behavior_vectors)
    pattern_library = get_pattern_library()
    return {
        "locations": locations,
        "device_events": device_events,
        "rules": rules,
        "alerts": alerts,
        "risk": risk,
        "coloc": coloc,
        "behavior_vectors": behavior_vectors,
        "hybrid_decisions": hybrid_decisions,
        "pattern_library": pattern_library,
    }


@st.cache_data(show_spinner=False)
def run_pipeline(participant_count: int, history_hours: int, point_interval: int) -> dict:
    cfg = SimulationConfig(
        participants=participant_count,
        duration_hours=history_hours,
        point_interval_min=point_interval,
    )
    locations = generate_locations(cfg)
    rules = generate_rules(locations["participant_id"].drop_duplicates().tolist())
    return build_dataset(locations, rules)


@st.cache_data(show_spinner=False)
def run_presentation_scenarios(participant_count: int) -> dict:
    background_participants = max(int(participant_count) - 3, 0)
    locations, rules = generate_presentation_scenarios(background_participants=background_participants)
    return build_dataset(locations, rules)


def pipeline_from_gpx(uploaded_files) -> dict | None:
    frames = []
    for file in uploaded_files:
        text = file.getvalue().decode("utf-8", errors="ignore")
        pid = participant_id_from_filename(file.name)
        frame = parse_gpx_text(text, participant_id=pid, city="Kaggle-EU")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return None

    locations = pd.concat(frames, ignore_index=True).sort_values("event_ts")
    rules = generate_rules(locations["participant_id"].drop_duplicates().tolist())
    return build_dataset(locations, rules)


def pipeline_from_template(template_file, participant_count: int) -> dict | None:
    if template_file is None:
        return None

    text = template_file.getvalue().decode("utf-8", errors="ignore")
    template = parse_gpx_text(text, participant_id="TEMPLATE", city="Template-EU")
    if template.empty:
        return None

    locations = generate_locations_from_template(template, participants=participant_count)
    rules = generate_rules(locations["participant_id"].drop_duplicates().tolist())
    return build_dataset(locations, rules)


def ensure_state(participant_options: list[str]) -> None:
    if "selected_participant" not in st.session_state or st.session_state["selected_participant"] not in participant_options:
        st.session_state["selected_participant"] = participant_options[0]
    if "focus_selected_on_map" not in st.session_state:
        st.session_state["focus_selected_on_map"] = False
    if "selected_anomaly_key" not in st.session_state:
        st.session_state["selected_anomaly_key"] = "None"
    if "applied_anomaly_focus_key" not in st.session_state:
        st.session_state["applied_anomaly_focus_key"] = "None"
    if "anomaly_focus_label" not in st.session_state:
        st.session_state["anomaly_focus_label"] = ""
    if "manual_incidents" not in st.session_state:
        st.session_state["manual_incidents"] = []
    if "incident_filter_mode" not in st.session_state:
        st.session_state["incident_filter_mode"] = "all"
    if "incident_filter_value" not in st.session_state:
        st.session_state["incident_filter_value"] = "All incidents"
    if "pending_selected_participant" not in st.session_state:
        st.session_state["pending_selected_participant"] = None
    if "pending_anomaly_focus_label" not in st.session_state:
        st.session_state["pending_anomaly_focus_label"] = None


def make_anomaly_table(alerts: pd.DataFrame, coloc: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "key": "None",
            "participant_id": "",
            "anomaly_type": "Clear focus",
            "event_ts": "",
            "subject": "",
            "counterpart": "",
            "severity": "",
            "details": "Reset anomaly-driven focus",
            "label": "No anomaly focus",
        }
    ]
    for idx, row in alerts.sort_values("event_ts", ascending=False).head(60).reset_index(drop=True).iterrows():
        rows.append(
            {
                "key": f"alert:{idx}",
                "participant_id": row["participant_id"],
                "anomaly_type": row["alert_type"],
                "event_ts": str(row["event_ts"]),
                "subject": row["participant_id"],
                "counterpart": "",
                "severity": row["severity"],
                "details": row["details"],
                "label": (
                    f"Alert | {row['event_ts']} | {row['participant_id']} | "
                    f"{row['alert_type']} | {row['severity']}"
                ),
            }
        )
    for idx, row in coloc.sort_values("event_ts", ascending=False).head(40).reset_index(drop=True).iterrows():
        details = f"Distance {float(row['distance_m']):.1f}m between {row['participant_a']} and {row['participant_b']}"
        rows.append(
            {
                "key": f"coloc:{idx}:a",
                "participant_id": row["participant_a"],
                "anomaly_type": "CO_LOCATION",
                "event_ts": str(row["event_ts"]),
                "subject": row["participant_a"],
                "counterpart": row["participant_b"],
                "severity": "HIGH",
                "details": details,
                "label": details + f" | focus {row['participant_a']}",
            }
        )
        rows.append(
            {
                "key": f"coloc:{idx}:b",
                "participant_id": row["participant_b"],
                "anomaly_type": "CO_LOCATION",
                "event_ts": str(row["event_ts"]),
                "subject": row["participant_b"],
                "counterpart": row["participant_a"],
                "severity": "HIGH",
                "details": details,
                "label": details + f" | focus {row['participant_b']}",
            }
        )
    return pd.DataFrame(rows)


def apply_anomaly_selection(anomaly_df: pd.DataFrame, selected_rows: list[int]) -> None:
    if not selected_rows:
        return
    anomaly_entry = anomaly_df.iloc[selected_rows[0]].to_dict()
    selected_anomaly_key = anomaly_entry["key"]
    if selected_anomaly_key == "None":
        st.session_state["selected_anomaly_key"] = "None"
        st.session_state["anomaly_focus_label"] = ""
        st.session_state["applied_anomaly_focus_key"] = "None"
        st.session_state["focus_selected_on_map"] = False
        return
    if selected_anomaly_key != st.session_state.get("applied_anomaly_focus_key"):
        st.session_state["selected_anomaly_key"] = selected_anomaly_key
        st.session_state["applied_anomaly_focus_key"] = selected_anomaly_key
        request_participant_focus(anomaly_entry["participant_id"], label=anomaly_entry["label"])
        st.rerun()






SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
DECISION_BONUS = {"ELEVATE": 30, "REVIEW": 15, "DEPRIORITIZE": -10, "NO_ACTION": -20}


def build_incident_queue(alerts: pd.DataFrame, coloc: pd.DataFrame, hybrid_decisions: pd.DataFrame, manual_incidents: list[dict] | None = None) -> pd.DataFrame:
    hybrid_lookup = hybrid_decisions.set_index("participant_id") if not hybrid_decisions.empty else pd.DataFrame()
    rows = []

    for idx, row in alerts.sort_values("event_ts", ascending=False).reset_index(drop=True).iterrows():
        hybrid = hybrid_lookup.loc[row["participant_id"]] if not hybrid_lookup.empty and row["participant_id"] in hybrid_lookup.index else None
        decision = hybrid["decision"] if hybrid is not None else "REVIEW"
        matched_pattern = hybrid["matched_pattern"] if hybrid is not None else "N/A"
        severity = row["severity"]
        if severity == "HIGH" and decision == "ELEVATE":
            severity = "CRITICAL"
        rows.append(
            {
                "key": f"alert:{idx}",
                "participant_id": row["participant_id"],
                "event_ts": row["event_ts"],
                "incident_type": row["alert_type"],
                "subject": row["participant_id"],
                "counterpart": "",
                "severity": severity,
                "decision": decision,
                "pattern": matched_pattern,
                "details": row["details"],
                "priority_score": SEVERITY_RANK[severity] * 100 + DECISION_BONUS.get(decision, 0) - idx,
                "label": (
                    f"{severity} | {row['alert_type']} | {row['participant_id']} | "
                    f"{decision} | {row['details']}"
                ),
            }
        )

    for idx, row in coloc.sort_values("event_ts", ascending=False).reset_index(drop=True).iterrows():
        decision_a = (
            hybrid_lookup.loc[row["participant_a"], "decision"]
            if not hybrid_lookup.empty and row["participant_a"] in hybrid_lookup.index
            else "REVIEW"
        )
        decision_b = (
            hybrid_lookup.loc[row["participant_b"], "decision"]
            if not hybrid_lookup.empty and row["participant_b"] in hybrid_lookup.index
            else "REVIEW"
        )
        focus_pid = row["participant_a"] if DECISION_BONUS.get(decision_a, 0) >= DECISION_BONUS.get(decision_b, 0) else row["participant_b"]
        focus_decision = decision_a if focus_pid == row["participant_a"] else decision_b
        severity = "CRITICAL" if "ELEVATE" in {decision_a, decision_b} else "HIGH"
        details = f"Distance {float(row['distance_m']):.1f}m between {row['participant_a']} and {row['participant_b']}"
        rows.append(
            {
                "key": f"coloc:{idx}:{'a' if focus_pid == row['participant_a'] else 'b'}",
                "participant_id": focus_pid,
                "event_ts": row["event_ts"],
                "incident_type": "CO_LOCATION",
                "subject": row["participant_a"],
                "counterpart": row["participant_b"],
                "severity": severity,
                "decision": focus_decision,
                "pattern": "Pair event",
                "details": details,
                "priority_score": SEVERITY_RANK[severity] * 100 + DECISION_BONUS.get(focus_decision, 0) - idx,
                "label": f"{severity} | CO_LOCATION | {row['participant_a']} ↔ {row['participant_b']} | focus {focus_pid}",
            }
        )

    for idx, incident in enumerate(manual_incidents or []):
        severity = incident.get("severity", "HIGH")
        decision = incident.get("decision", "ELEVATE")
        rows.append(
            {
                "key": incident.get("key", f"manual:{idx}"),
                "participant_id": incident.get("participant_id", ""),
                "event_ts": incident.get("event_ts", datetime.now(timezone.utc)),
                "incident_type": incident.get("incident_type", "MANUAL_TEST"),
                "subject": incident.get("subject", incident.get("participant_id", "")),
                "counterpart": incident.get("counterpart", ""),
                "severity": severity,
                "decision": decision,
                "pattern": incident.get("pattern", "Manual Drill"),
                "details": incident.get("details", "Manual operator drill"),
                "priority_score": SEVERITY_RANK.get(severity, 1) * 100 + DECISION_BONUS.get(decision, 0) + 50,
                "label": incident.get("label", f"{severity} | {incident.get('incident_type', 'MANUAL_TEST')} | {incident.get('participant_id', '')}"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "key", "participant_id", "event_ts", "incident_type", "subject", "counterpart",
                "severity", "decision", "pattern", "details", "priority_score", "label"
            ]
        )

    incident_df = pd.DataFrame(rows)
    incident_df = incident_df.sort_values(["priority_score", "event_ts"], ascending=[False, False]).reset_index(drop=True)
    return incident_df




def render_escalation_banners(incident_df: pd.DataFrame) -> None:
    banner_df = incident_df[incident_df["severity"].isin(["CRITICAL", "HIGH"])].head(3)
    if banner_df.empty:
        return

    st.subheader("Operator Escalations")
    for _, row in banner_df.iterrows():
        tone = "#7f1d1d" if row["severity"] == "CRITICAL" else "#78350f"
        border = "#ef4444" if row["severity"] == "CRITICAL" else "#f59e0b"
        counterpart = f" · Counterpart: {row['counterpart']}" if row["counterpart"] else ""
        html = (
            f"<div style='padding:14px 16px;border-radius:10px;margin-bottom:10px;"
            f"border-left:8px solid {border};background:{tone};'>"
            f"<strong>{row['severity']} · {row['incident_type']}</strong><br/>"
            f"Subject: {row['subject']}{counterpart}<br/>"
            f"Decision: {row['decision']} · Pattern: {row['pattern']}<br/>"
            f"{row['details']}"
            f"</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


def render_test_trigger_panel(participant_options: list[str]) -> None:
    st.subheader("Trigger Test Incident")
    st.caption("Create visible escalation drills and verify operator workflows.")
    with st.form("manual_incident_form", clear_on_submit=False):
        participant_id = st.selectbox("Subject", participant_options, key="manual_trigger_participant")
        incident_type = st.selectbox("Incident Type", ["OFFICER_DOWN", "TAMPER", "NO_SIGNAL", "PANIC_BUTTON", "MANUAL_TEST"], index=0)
        severity = st.selectbox("Severity", ["CRITICAL", "HIGH", "MEDIUM"], index=0)
        details = st.text_input("Operator Note", value="Manual escalation drill triggered by operator")
        submitted = st.form_submit_button("Trigger Incident", use_container_width=True)
        if submitted:
            decision = "ELEVATE" if severity in {"CRITICAL", "HIGH"} else "REVIEW"
            incident = {
                "key": f"manual:{datetime.now(timezone.utc).timestamp()}",
                "participant_id": participant_id,
                "event_ts": datetime.now(timezone.utc),
                "incident_type": incident_type,
                "subject": participant_id,
                "counterpart": "",
                "severity": severity,
                "decision": decision,
                "pattern": "Manual Drill",
                "details": details,
                "label": f"{severity} | {incident_type} | {participant_id} | {decision}",
            }
            st.session_state["manual_incidents"] = [incident] + st.session_state.get("manual_incidents", [])[:19]
            request_participant_focus(participant_id, label=incident["label"])
            st.rerun()
    cols = st.columns(2)
    if cols[0].button("Trigger Critical Drill", use_container_width=True):
        participant_id = st.session_state.get("selected_participant", participant_options[0])
        incident = {
            "key": f"manual:{datetime.now(timezone.utc).timestamp()}",
            "participant_id": participant_id,
            "event_ts": datetime.now(timezone.utc),
            "incident_type": "CRITICAL_DRILL",
            "subject": participant_id,
            "counterpart": "",
            "severity": "CRITICAL",
            "decision": "ELEVATE",
            "pattern": "Manual Drill",
            "details": "Critical escalation drill triggered by operator",
            "label": f"CRITICAL | CRITICAL_DRILL | {participant_id} | ELEVATE",
        }
        st.session_state["manual_incidents"] = [incident] + st.session_state.get("manual_incidents", [])[:19]
        request_participant_focus(participant_id, label=incident["label"])
        st.rerun()
    if cols[1].button("Clear Test Incidents", use_container_width=True):
        st.session_state["manual_incidents"] = []
        st.rerun()





def request_participant_focus(participant_id: str, label: str | None = None, clear_anomaly: bool = False) -> None:
    st.session_state["pending_selected_participant"] = participant_id
    st.session_state["focus_selected_on_map"] = True
    if clear_anomaly:
        st.session_state["pending_anomaly_focus_label"] = ""
        st.session_state["applied_anomaly_focus_key"] = "None"
    elif label is not None:
        st.session_state["pending_anomaly_focus_label"] = label


def apply_pending_participant_focus() -> None:
    pending = st.session_state.get("pending_selected_participant")
    if not pending:
        return
    st.session_state["selected_participant"] = pending
    pending_label = st.session_state.get("pending_anomaly_focus_label")
    if pending_label is not None:
        st.session_state["anomaly_focus_label"] = pending_label
        st.session_state["pending_anomaly_focus_label"] = None
    st.session_state["pending_selected_participant"] = None

def set_incident_filter(mode: str, value: str) -> None:
    st.session_state["incident_filter_mode"] = mode
    st.session_state["incident_filter_value"] = value


def filter_incident_queue(incident_df: pd.DataFrame) -> pd.DataFrame:
    mode = st.session_state.get("incident_filter_mode", "all")
    value = st.session_state.get("incident_filter_value", "All incidents")
    if incident_df.empty or mode == "all":
        return incident_df
    if mode == "severity":
        return incident_df[incident_df["severity"] == value].reset_index(drop=True)
    if mode == "decision":
        return incident_df[incident_df["decision"] == value].reset_index(drop=True)
    return incident_df


def get_incident_scope_participants(filtered_incidents: pd.DataFrame) -> set[str]:
    if filtered_incidents.empty:
        return set()
    participants = set(filtered_incidents["participant_id"].dropna().astype(str).tolist())
    participants.update(filtered_incidents["subject"].dropna().astype(str).tolist())
    participants.update(
        value for value in filtered_incidents["counterpart"].dropna().astype(str).tolist() if value
    )
    return {value for value in participants if value}

def render_incident_overview(incident_df: pd.DataFrame) -> None:
    st.subheader("Incident Overview")
    if incident_df.empty:
        st.info("No active incidents in the current dataset.")
        return

    total_open = len(incident_df)
    critical = int((incident_df["severity"] == "CRITICAL").sum())
    high = int((incident_df["severity"] == "HIGH").sum())
    elevated = int((incident_df["decision"] == "ELEVATE").sum())
    review = int((incident_df["decision"] == "REVIEW").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Incidents", f"{total_open:,}")
    if c1.button("Show all", key="overview_all", use_container_width=True):
        set_incident_filter("all", "All incidents")
        st.rerun()

    c2.metric("Critical", f"{critical:,}")
    if c2.button("Focus critical", key="overview_critical", use_container_width=True):
        set_incident_filter("severity", "CRITICAL")
        st.rerun()

    c3.metric("High", f"{high:,}")
    if c3.button("Focus high", key="overview_high", use_container_width=True):
        set_incident_filter("severity", "HIGH")
        st.rerun()

    c4.metric("Escalate", f"{elevated:,}")
    if c4.button("Focus escalate", key="overview_escalate", use_container_width=True):
        set_incident_filter("decision", "ELEVATE")
        st.rerun()

    c5.metric("Review", f"{review:,}")
    if c5.button("Focus review", key="overview_review", use_container_width=True):
        set_incident_filter("decision", "REVIEW")
        st.rerun()

    c6.metric("Active Filter", st.session_state.get("incident_filter_value", "All incidents"))
    if c6.button("Clear filter", key="overview_clear_filter", use_container_width=True):
        set_incident_filter("all", "All incidents")
        st.rerun()

def format_hybrid_view(hybrid_decisions: pd.DataFrame) -> pd.DataFrame:
    view = hybrid_decisions.copy()
    if view.empty:
        return view
    view["similarity_score"] = view["similarity_score"].apply(
        lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
    )
    return view

def render_global_metrics(dataset: dict) -> None:
    locations = dataset["locations"]
    alerts = dataset["alerts"]
    coloc = dataset["coloc"]
    hybrid = dataset["hybrid_decisions"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Participants", f"{locations['participant_id'].nunique()}")
    m2.metric("Location Events", f"{len(locations):,}")
    m3.metric("Alerts", f"{len(alerts):,}")
    m4.metric("Co-location", f"{len(coloc):,}")
    m5.metric("Elevated", f"{int((hybrid['decision'] == 'ELEVATE').sum()):,}")
    m6.metric("Deprioritized", f"{int((hybrid['decision'] == 'DEPRIORITIZE').sum()):,}")


def render_overview_banner(scenario_mode: str, workspace: str) -> None:
    c1, c2 = st.columns([1.6, 1])
    c1.caption(f"Dataset: `{scenario_mode}`")
    c2.caption(f"Workspace: `{workspace}`")


def render_live_map(latest: pd.DataFrame, *, title: str, key: str, focus_selected: bool = True) -> None:
    latest = latest.copy()
    latest["marker_radius"] = latest["participant_id"].apply(
        lambda participant_id: 26000 if participant_id == st.session_state["selected_participant"] else 18000
    )
    latest["fill_color"] = latest["participant_id"].apply(
        lambda participant_id: [255, 196, 0, 235]
        if participant_id == st.session_state["selected_participant"]
        else [220, 60, 50, 180]
    )

    st.subheader(title)
    selected_latest = latest[latest["participant_id"] == st.session_state["selected_participant"]]
    lat_span = max(float(latest["lat"].max() - latest["lat"].min()), 0.8)
    lon_span = max(float(latest["lon"].max() - latest["lon"].min()), 0.8)
    center_lat = float(latest["lat"].mean())
    center_lon = float(latest["lon"].mean())
    max_span = max(lat_span, lon_span)
    default_zoom = min(6.2, max(3.2, 7.6 - math.log(max_span + 1.0, 2)))
    if focus_selected and st.session_state["focus_selected_on_map"] and not selected_latest.empty:
        focus_row = selected_latest.iloc[0]
        center_lat = float(focus_row["lat"])
        center_lon = float(focus_row["lon"])
        zoom = 10.5
    else:
        zoom = default_zoom

    map_event = st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=20),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    id="latest-positions",
                    data=latest,
                    get_position="[lon, lat]",
                    get_radius="marker_radius",
                    radius_min_pixels=8,
                    radius_max_pixels=30,
                    get_fill_color="fill_color",
                    get_line_color=[255, 255, 255, 220],
                    line_width_min_pixels=1,
                    stroked=True,
                    pickable=True,
                    auto_highlight=True,
                )
            ],
            tooltip={"text": "Participant: {participant_id}\nCity: {city}\nLat/Lon: {lat}, {lon}"},
        ),
        width=None,
        height=480,
        on_select="rerun",
        selection_mode="single-object",
        key=key,
    )
    selected_objects = map_event.get("selection", {}).get("objects", {}).get("latest-positions", []) if map_event else []
    if selected_objects:
        clicked_participant = selected_objects[0].get("participant_id")
        if clicked_participant:
            request_participant_focus(clicked_participant, clear_anomaly=True)
            st.rerun()


def get_subject_frames(dataset: dict, selected: str) -> dict:
    return {
        "alerts": dataset["alerts"][dataset["alerts"]["participant_id"] == selected].sort_values("event_ts", ascending=False),
        "track": dataset["locations"][dataset["locations"]["participant_id"] == selected].sort_values("event_ts", ascending=False),
        "device": dataset["device_events"][dataset["device_events"]["participant_id"] == selected].sort_values("event_ts", ascending=False),
        "rules": dataset["rules"][dataset["rules"]["participant_id"] == selected],
        "risk": dataset["risk"][dataset["risk"]["participant_id"] == selected].sort_values("risk_score", ascending=False),
        "behavior": dataset["behavior_vectors"][dataset["behavior_vectors"]["participant_id"] == selected],
        "hybrid": dataset["hybrid_decisions"][dataset["hybrid_decisions"]["participant_id"] == selected],
    }


def render_subject_selector(participant_options: list[str], label: str = "Selected subject") -> str:
    selected = st.selectbox(
        label,
        participant_options,
        key="selected_participant",
        help="Click a marker on the map, an anomaly row, or choose a participant here.",
    )
    st.session_state["focus_selected_on_map"] = True
    return selected


def render_subject_summary(selected: str, frames: dict) -> None:
    latest_position = frames["track"].iloc[0] if not frames["track"].empty else None
    latest_device = frames["device"].iloc[0] if not frames["device"].empty else None
    risk_row = frames["risk"].iloc[0] if not frames["risk"].empty else None
    rule_row = frames["rules"].iloc[0] if not frames["rules"].empty else None
    behavior_row = frames["behavior"].iloc[0] if not frames["behavior"].empty else None
    hybrid_row = frames["hybrid"].iloc[0] if not frames["hybrid"].empty else None

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Participant", selected)
    d2.metric("Active Alerts", f"{len(frames['alerts']):,}")
    d3.metric("Risk", risk_row["risk_band"] if risk_row is not None else "N/A", f"{risk_row['risk_score']:.1f}" if risk_row is not None else None)
    d4.metric(
        "Battery",
        f"{float(latest_device['battery_pct']):.0f}%" if latest_device is not None else "N/A",
        "Tamper" if latest_device is not None and int(latest_device["tamper_flag"]) == 1 else "Normal",
    )
    d5.metric("Hybrid", hybrid_row["decision"] if hybrid_row is not None else "N/A", hybrid_row["matched_pattern"] if hybrid_row is not None else None)

    st.json(
        {
            "city": latest_position["city"] if latest_position is not None else "N/A",
            "last_seen": str(latest_position["event_ts"]) if latest_position is not None else "N/A",
            "speed_kmh": round(float(latest_position["speed_kmh"]), 2) if latest_position is not None else "N/A",
            "coordinates": (
                f"{float(latest_position['lat']):.6f}, {float(latest_position['lon']):.6f}"
                if latest_position is not None
                else "N/A"
            ),
            "zone_rule": rule_row["zone_name"] if rule_row is not None else "N/A",
            "behavior_vector": behavior_row["behavior_vector"] if behavior_row is not None else "N/A",
            "pattern_match": hybrid_row["matched_pattern"] if hybrid_row is not None else "N/A",
            "decision": hybrid_row["decision"] if hybrid_row is not None else "N/A",
        },
        expanded=False,
    )


def render_copilot(settings: dict, selected: str, frames: dict) -> None:
    latest_device = frames["device"].iloc[0] if not frames["device"].empty else None
    risk_row = frames["risk"].iloc[0] if not frames["risk"].empty else None
    rule_row = frames["rules"].iloc[0] if not frames["rules"].empty else None
    hybrid_row = frames["hybrid"].iloc[0] if not frames["hybrid"].empty else None
    behavior_row = frames["behavior"].iloc[0] if not frames["behavior"].empty else None

    st.subheader("Officer Copilot")
    st.caption(f"Provider: `{settings['llm_provider']}`")
    context = {
        "participant_id": selected,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alerts": frames["alerts"].head(15).to_dict(orient="records"),
        "latest_track": frames["track"].head(20).to_dict(orient="records"),
        "latest_device": latest_device.to_dict() if latest_device is not None else {},
        "risk": risk_row.to_dict() if risk_row is not None else {},
        "hybrid_decision": hybrid_row.to_dict() if hybrid_row is not None else {},
        "behavior_vector": behavior_row.to_dict() if behavior_row is not None else {},
        "rule": rule_row.to_dict() if rule_row is not None else {},
    }

    if latest_device is not None:
        st.write(
            f"Latest device status: battery {float(latest_device['battery_pct']):.0f}%, "
            f"tamper={int(latest_device['tamper_flag'])}, no_signal={int(latest_device['no_signal_flag'])}"
        )
    if risk_row is not None:
        st.write(f"Risk score: {float(risk_row['risk_score']):.1f} ({risk_row['risk_band']})")
    if hybrid_row is not None:
        st.write(f"Hybrid decision: {hybrid_row['decision']} via {hybrid_row['matched_pattern']} (score={hybrid_row['similarity_score']})")

    t1, t2 = st.columns(2)
    if t1.button("Test LLM"):
        ok, _ = explain_alert('{"healthcheck":"ok"}', settings), ""
        if "failed" in ok.lower() and "error:" in ok.lower():
            st.error(ok)
        else:
            st.success("LLM reachable.")
    if t2.button("Explain latest situation"):
        with st.spinner("Generating explanation..."):
            answer = explain_alert(json.dumps(context, default=str), settings)
        st.write(answer)

    st.code(json.dumps(context, default=str, indent=2)[:6000], language="json")


def render_operations_center(dataset: dict, participant_options: list[str], settings: dict) -> None:
    latest = dataset["locations"][dataset["locations"]["event_ts"] == dataset["locations"]["event_ts"].max()].copy()
    incident_df = build_incident_queue(
        dataset["alerts"],
        dataset["coloc"],
        dataset["hybrid_decisions"],
        manual_incidents=st.session_state.get("manual_incidents", []),
    )
    filtered_incidents = filter_incident_queue(incident_df)
    scoped_participants = get_incident_scope_participants(filtered_incidents)
    if scoped_participants:
        filtered_latest = latest[latest["participant_id"].isin(scoped_participants)].copy()
        if filtered_latest.empty:
            filtered_latest = latest.copy()
    else:
        filtered_latest = latest.copy()

    render_incident_overview(incident_df)
    render_escalation_banners(filtered_incidents if not filtered_incidents.empty else incident_df)

    top_left, top_mid, top_right = st.columns([1.35, 1, 1])
    with top_left:
        render_live_map(filtered_latest, title="Live Map", key="ops_live_map", focus_selected=not bool(scoped_participants))
        if st.session_state.get("incident_filter_mode", "all") != "all":
            st.caption(f"Map scope: `{st.session_state.get('incident_filter_value', 'All incidents')}`")
    with top_mid:
        st.subheader("Prioritized Incident Queue")
        st.caption("Incidents are ordered by severity first, then hybrid decision urgency.")
        incident_event = st.dataframe(
            filtered_incidents[["severity", "decision", "incident_type", "event_ts", "subject", "counterpart", "details"]],
            width="stretch",
            height=480,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="ops_incident_table",
        )
        apply_anomaly_selection(filtered_incidents, incident_event.get("selection", {}).get("rows", []) if incident_event else [])
        if st.session_state.get("anomaly_focus_label"):
            st.info(f"Current focus: {st.session_state['anomaly_focus_label']}")
        render_subject_selector(participant_options, label="Quick subject focus")
    with top_right:
        render_test_trigger_panel(participant_options)

    render_operator_attention_feed(filtered_incidents if not filtered_incidents.empty else incident_df, dataset["hybrid_decisions"], settings)

    st.subheader("Current Workload")
    tab1, tab2, tab3, tab4 = st.tabs(["Incidents", "Alerts", "Co-location", "Hybrid Decisions"])
    with tab1:
        st.dataframe(filtered_incidents, width="stretch", hide_index=True)
    with tab2:
        st.dataframe(dataset["alerts"].sort_values("event_ts", ascending=False).head(100), width="stretch")
    with tab3:
        st.dataframe(dataset["coloc"].sort_values("event_ts", ascending=False).head(100), width="stretch")
    with tab4:
        st.dataframe(format_hybrid_view(dataset["hybrid_decisions"]), width="stretch", hide_index=True)




def build_operator_watch_context(incident_df: pd.DataFrame, hybrid_decisions: pd.DataFrame) -> tuple[str, str]:
    top_incidents = incident_df.head(8).copy() if not incident_df.empty else incident_df.copy()
    payload = {
        "top_incidents": top_incidents[["severity", "decision", "incident_type", "event_ts", "subject", "counterpart", "details"]].to_dict(orient="records") if not top_incidents.empty else [],
        "decision_summary": hybrid_decisions["decision"].value_counts(dropna=False).to_dict() if not hybrid_decisions.empty else {},
    }
    signature = json.dumps(payload, default=str, sort_keys=True)
    return signature, json.dumps(payload, default=str, indent=2)


def render_operator_attention_feed(incident_df: pd.DataFrame, hybrid_decisions: pd.DataFrame, settings: dict) -> None:
    st.subheader("Operator Attention Feed")
    st.caption("LLM triage highlights incidents that operators should review first.")
    if incident_df.empty:
        st.info("No active incidents to triage.")
        return

    signature, context_text = build_operator_watch_context(incident_df, hybrid_decisions)
    if st.session_state.get("operator_watch_signature") != signature:
        st.session_state["operator_watch_signature"] = signature
        st.session_state["operator_watch_text"] = triage_incidents(context_text, config=settings)

    if st.button("Refresh LLM Triage", key="refresh_llm_triage"):
        st.session_state["operator_watch_signature"] = signature
        st.session_state["operator_watch_text"] = triage_incidents(context_text, config=settings)

    attention_text = st.session_state.get("operator_watch_text", "")
    if attention_text:
        lowered = attention_text.lower()
        if "failed" in lowered and "error:" in lowered:
            st.error(attention_text)
        else:
            st.warning(attention_text)
    else:
        st.info("LLM triage will appear here when incidents are available.")


def render_subject_investigation(dataset: dict, participant_options: list[str], settings: dict) -> None:
    latest = dataset["locations"][dataset["locations"]["event_ts"] == dataset["locations"]["event_ts"].max()].copy()
    subject = render_subject_selector(participant_options)
    frames = get_subject_frames(dataset, subject)

    if st.session_state.get("anomaly_focus_label"):
        st.caption(f"Focused from anomaly: {st.session_state['anomaly_focus_label']}")
    else:
        st.caption("Use the map, anomaly queue, or selector to pick a subject.")

    map_col, detail_col = st.columns([1.2, 1])
    with map_col:
        render_live_map(latest, title="Subject Focus Map", key="investigation_map")
    with detail_col:
        render_subject_summary(subject, frames)

    lower_left, lower_right = st.columns([1.2, 1])
    with lower_left:
        st.subheader("Recent Alerts")
        st.dataframe(frames["alerts"].head(25), width="stretch")
        st.subheader("Recent Track")
        st.dataframe(frames["track"][["event_ts", "city", "lat", "lon", "speed_kmh"]].head(25), width="stretch")
    with lower_right:
        render_copilot(settings, subject, frames)


def render_hybrid_intelligence(dataset: dict) -> None:
    st.subheader("Converged Spatial + Vector Decisions")
    st.caption("Geofence detection + behavioral vectors + reference-pattern matching.")
    st.dataframe(format_hybrid_view(dataset["hybrid_decisions"]), width="stretch", hide_index=True)

    upper_left, upper_right = st.columns(2)
    with upper_left:
        st.subheader("Behavior Vectors")
        st.dataframe(
            dataset["behavior_vectors"][["participant_id", "zone_name", "speed_score", "dwell_score", "proximity_score", "behavior_vector"]],
            width="stretch",
            hide_index=True,
        )
    with upper_right:
        st.subheader("Reference Patterns")
        st.dataframe(
            dataset["pattern_library"][["pattern_id", "description", "decision_hint", "behavior_vector"]],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Decision Outcome Summary")
    summary = dataset["hybrid_decisions"]["decision"].value_counts(dropna=False).rename_axis("decision").reset_index(name="subjects")
    st.dataframe(summary, width="stretch", hide_index=True)


def render_data_and_scenarios(dataset: dict, scenario_mode: str) -> None:
    st.subheader("Scenario Workspace")
    st.caption("Use the sidebar to switch sources, participant counts, and GPX/template inputs. This view helps inspect the generated dataset.")

    if scenario_mode == "Presentation scenarios":
        st.info("Presentation mode is active: routine drive-by, loitering risk, and outside-zone no-action scenarios are preloaded.")

    t1, t2, t3, t4 = st.tabs(["Locations", "Rules", "Devices", "Risk Ranking"])
    with t1:
        st.dataframe(dataset["locations"].sort_values(["participant_id", "event_ts"], ascending=[True, False]).head(300), width="stretch")
    with t2:
        st.dataframe(dataset["rules"], width="stretch", hide_index=True)
    with t3:
        st.dataframe(dataset["device_events"].sort_values(["participant_id", "event_ts"], ascending=[True, False]).head(300), width="stretch")
    with t4:
        st.dataframe(dataset["risk"].sort_values("risk_score", ascending=False), width="stretch", hide_index=True)


def render_oracle_lab(dataset: dict, settings: dict) -> None:
    st.subheader("Oracle 26ai Converged Demo")
    st.caption("Set up Spatial + Vector tables, load the current dataset, and run the Oracle hybrid query.")
    repo = OracleRepository(
        user=settings["db_user"],
        password=settings["db_password"],
        dsn=settings["db_dsn"],
    )

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Test DB"):
        ok, msg = repo.test_connection()
        (st.success if ok else st.error)(msg)
    if c2.button("Setup Converged Schema"):
        try:
            repo.setup_converged_schema()
            st.success("Converged schema created in Oracle.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Schema setup failed: {exc}")
    if c3.button("Push Hybrid Demo To Oracle"):
        try:
            repo.load_converged_demo(dataset["locations"], dataset["rules"], dataset["pattern_library"], dataset["behavior_vectors"])
            st.success("Current demo data loaded into Oracle converged tables.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Oracle load failed: {exc}")
    if c4.button("Run Oracle Hybrid Query"):
        try:
            st.session_state["oracle_hybrid_results"] = repo.run_hybrid_decisions()
            st.success("Oracle hybrid query completed.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Oracle query failed: {exc}")

    ot1, ot2, ot3 = st.tabs(["Hybrid SQL Results", "Baseline Persistence", "Schema Script"])
    with ot1:
        if "oracle_hybrid_results" in st.session_state:
            st.dataframe(st.session_state["oracle_hybrid_results"], width="stretch", hide_index=True)
        else:
            st.info("Run `Run Oracle Hybrid Query` to view converged SQL results.")
    with ot2:
        st.caption("Push the baseline persistence tables used by the broader app.")
        if st.button("Push Baseline Tables To Oracle"):
            if not repo.configured():
                st.warning("Database settings are incomplete.")
            else:
                repo.write_dataframe(dataset["locations"], "LOCATION_EVENTS")
                repo.write_dataframe(dataset["device_events"], "DEVICE_EVENTS")
                repo.write_dataframe(
                    dataset["rules"][["participant_id", "rule_type", "zone_name", "zone_lat", "zone_lon", "radius_m"]],
                    "RULES_SCHEDULES",
                )
                repo.write_dataframe(dataset["alerts"], "ALERTS")
                repo.write_dataframe(dataset["risk"], "RISK_SCORES")
                st.success("Baseline data pushed to Oracle.")
    with ot3:
        st.code(repo.script_text("sql/schema_oracle_26ai_converged.sql"), language="sql")


with st.sidebar:
    st.header("Workspace")
    workspace = st.radio("View", WORKSPACES, index=0)

    st.divider()
    with st.expander("Scenario Controls", expanded=True):
        scenario_mode = st.radio(
            "Data source",
            ["Presentation scenarios", "EU synthetic", "Template-based synthetic", "GPX upload"],
            index=0,
        )
        participants = st.slider("Participants", min_value=3, max_value=120, value=35, step=1)
        duration_hours = st.slider("Hours of history", min_value=2, max_value=24, value=8, step=1)
        interval = st.selectbox("Point interval (minutes)", [1, 2, 5, 10], index=2)
        template_gpx = st.file_uploader(
            "Template GPX",
            accept_multiple_files=False,
            type=["gpx"],
            disabled=scenario_mode != "Template-based synthetic",
        )
        gpx_files = st.file_uploader(
            "GPX Upload",
            accept_multiple_files=True,
            type=["gpx"],
            disabled=scenario_mode != "GPX upload",
        )
        run = st.button("Generate / Refresh", use_container_width=True)

    with st.expander("Runtime Settings", expanded=False):
        llm_provider = st.selectbox("LLM Provider", ["ollama", "oci"], index=0)
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        ollama_model = st.text_input("Ollama Model", value="gpt-oss:20b")
        oci_model_id = st.text_input("OCI Model ID", value="")
        oci_compartment_ocid = st.text_input("OCI Compartment OCID", value="")
        oci_config_file = st.text_input("OCI Config File", value="~/.oci/config")
        oci_config_profile = st.text_input("OCI Config Profile", value="DEFAULT")
        db_user = st.text_input("Oracle User", value="sentinel")
        db_password = st.text_input("Oracle Password", value="SentinelPwd123", type="password")
        db_dsn = st.text_input("Oracle DSN", value="localhost:1521/FREEPDB1")

settings = {
    "llm_provider": llm_provider,
    "ollama_url": ollama_url,
    "ollama_model": ollama_model,
    "oci_model_id": oci_model_id,
    "oci_compartment_ocid": oci_compartment_ocid,
    "oci_config_file": oci_config_file,
    "oci_config_profile": oci_config_profile,
    "db_user": db_user,
    "db_password": db_password,
    "db_dsn": db_dsn,
}

if run or "data_loaded" not in st.session_state:
    if scenario_mode == "GPX upload":
        dataset = pipeline_from_gpx(gpx_files) if gpx_files else None
        if dataset is None:
            st.warning("GPX upload parsed no points. Falling back to synthetic EU data.")
            dataset = run_pipeline(participants, duration_hours, interval)
    elif scenario_mode == "Template-based synthetic":
        dataset = pipeline_from_template(template_gpx, participant_count=participants)
        if dataset is None:
            st.warning("Template GPX parsed no points. Falling back to synthetic EU data.")
            dataset = run_pipeline(participants, duration_hours, interval)
    elif scenario_mode == "Presentation scenarios":
        dataset = run_presentation_scenarios(participants)
    else:
        dataset = run_pipeline(participants, duration_hours, interval)
    st.session_state["data"] = dataset
    st.session_state["data_loaded"] = True
    st.session_state["scenario_mode"] = scenario_mode
    st.session_state.pop("oracle_hybrid_results", None)


dataset = st.session_state["data"]
participant_options = sorted(dataset["locations"]["participant_id"].unique().tolist())
ensure_state(participant_options)
apply_pending_participant_focus()
render_overview_banner(st.session_state.get("scenario_mode", scenario_mode), workspace)
render_global_metrics(dataset)

if workspace == "Operations Center":
    render_operations_center(dataset, participant_options, settings)
elif workspace == "Subject Investigation":
    render_subject_investigation(dataset, participant_options, settings)
elif workspace == "Hybrid Intelligence":
    render_hybrid_intelligence(dataset)
elif workspace == "Data & Scenarios":
    render_data_and_scenarios(dataset, st.session_state.get("scenario_mode", scenario_mode))
else:
    render_oracle_lab(dataset, settings)
