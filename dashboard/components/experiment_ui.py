from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def _download_bytes(payload: str, filename: str, mime_type: str) -> None:
    st.download_button(
        label=f"Download {filename}",
        data=payload,
        file_name=filename,
        mime=mime_type,
        use_container_width=True,
    )


def render_metrics_panel(result: dict[str, Any]) -> None:
    st.subheader("Experiment Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Execution Time", f"{result.get('execution_time', 0.0):.3f}s")
    metric_cols[1].metric("Planning Time", f"{result.get('planning_time', 0.0):.3f}s")
    metric_cols[2].metric("Safety Status", str(result.get("safety_status", "unknown")).upper())
    metric_cols[3].metric("Task Success", "Yes" if result.get("task_success") else "No")


def render_results_panel(result: dict[str, Any]) -> None:
    tab_names = ["Plan", "Safety", "Execution", "Reflection", "Summary", "Exports"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.json(result.get("generated_plan", {}))

    with tabs[1]:
        st.json(result.get("safety_report", {}))

    with tabs[2]:
        st.json(result.get("execution_status", {}))

    with tabs[3]:
        st.json(result.get("reflection_report", {}) or {})

    with tabs[4]:
        st.success(result.get("summary", "No summary available."))

    with tabs[5]:
        experiment_dir = Path("logs") / "experiments" / result.get("experiment_id", "")
        if experiment_dir.exists():
            log_text = (experiment_dir / "execution.log").read_text(encoding="utf-8") if (experiment_dir / "execution.log").exists() else ""
            csv_text = (experiment_dir / "summary.csv").read_text(encoding="utf-8") if (experiment_dir / "summary.csv").exists() else ""
            json_text = (experiment_dir / "experiment.json").read_text(encoding="utf-8") if (experiment_dir / "experiment.json").exists() else ""
            _download_bytes(log_text, f"{result.get('experiment_id', 'experiment')}.log", "text/plain")
            _download_bytes(json_text, f"{result.get('experiment_id', 'experiment')}.json", "application/json")
            _download_bytes(csv_text, f"{result.get('experiment_id', 'experiment')}.csv", "text/csv")
        else:
            st.info("No export files were generated for the current experiment.")


def render_history_panel(history: list[dict[str, Any]]) -> None:
    st.subheader("Experiment History")
    if not history:
        st.info("No experiment history yet.")
        return

    frame = pd.DataFrame(history)
    display_columns = ["timestamp", "instruction", "provider", "safety_status", "task_success", "total_time"]
    st.dataframe(frame[display_columns], use_container_width=True, hide_index=True)

    if st.checkbox("Inspect a saved experiment", value=False):
        selected = st.selectbox("Select experiment", options=[item.get("experiment_id") for item in history if item.get("experiment_id")])
        for item in history:
            if item.get("experiment_id") == selected:
                st.json(item)
                break


def render_visualizations(history_frame: pd.DataFrame) -> None:
    st.subheader("Research Visualizations")
    if history_frame.empty:
        st.info("Run at least one experiment to populate the plots.")
        return

    success_timeline = px.scatter(
        history_frame,
        x="timestamp",
        y="task_success",
        color="task_success",
        symbol="safety_status",
        title="Success Timeline",
        labels={"task_success": "Successful", "timestamp": "Timestamp"},
    )
    success_timeline.update_yaxes(tickvals=[0, 1], ticktext=["Failed", "Succeeded"])
    st.plotly_chart(success_timeline, use_container_width=True)

    latency_chart = px.bar(
        history_frame,
        x="timestamp",
        y="total_time",
        color="task_success",
        title="Latency Chart",
        labels={"total_time": "Total Time (s)", "timestamp": "Timestamp"},
    )
    st.plotly_chart(latency_chart, use_container_width=True)

    safety_counts = history_frame["safety_status"].value_counts().reset_index()
    safety_counts.columns = ["safety_status", "count"]
    safety_chart = px.bar(safety_counts, x="safety_status", y="count", title="Safety Violations")
    st.plotly_chart(safety_chart, use_container_width=True)

    failure_types = []
    for item in history_frame.to_dict(orient="records"):
        if item.get("failure_report") and item.get("failure_report", {}).get("failure_type"):
            failure_types.append(item["failure_report"]["failure_type"])
    if failure_types:
        failure_frame = pd.DataFrame({"failure_type": failure_types})
        failure_chart = px.histogram(failure_frame, x="failure_type", title="Failure Types")
        st.plotly_chart(failure_chart, use_container_width=True)
    else:
        st.info("No failures recorded yet.")

    recovery_counts = []
    for item in history_frame.to_dict(orient="records"):
        if item.get("recovery_decision"):
            recovery_counts.append(item["recovery_decision"].get("strategy", "none"))
    if recovery_counts:
        recovery_frame = pd.DataFrame({"strategy": recovery_counts})
        recovery_chart = px.histogram(recovery_frame, x="strategy", title="Recovery Attempts")
        st.plotly_chart(recovery_chart, use_container_width=True)
    else:
        st.info("No recovery actions recorded yet.")
