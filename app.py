from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor.executor import ExecutionEngine
from executor.logger import ExecutionLogger
from llm.config import LLMConfig, load_config
from llm.factory import create_provider
from planner.planner import PlannerAgent
from safety.safety_guard import SafetyGuard
from self_correction.failure_detector import FailureDetector
from self_correction.reflection import ReflectionEngine
from simulator.environment import MetaWorldEnvironment

from dashboard.components.experiment_ui import (
    render_history_panel,
    render_metrics_panel,
    render_results_panel,
    render_visualizations,
)


st.set_page_config(page_title="SafeLite Research Dashboard", page_icon="🧪", layout="wide")


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())
    if hasattr(value, "dict"):
        return _serialize(value.dict())
    return value


def _load_history() -> list[dict[str, Any]]:
    history_path = ROOT / "logs" / "experiments" / "history.json"
    if not history_path.exists():
        return []
    try:
        return json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_history(history: list[dict[str, Any]]) -> None:
    history_path = ROOT / "logs" / "experiments" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _run_experiment(instruction: str, provider_name: str, api_key: str) -> dict[str, Any]:
    experiment_dir = ROOT / "logs" / "experiments" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config()
    provider_config = LLMConfig(
        provider=provider_name,
        api_key=api_key or base_config.api_key,
        model_name=base_config.model_name,
        temperature=base_config.temperature,
        max_tokens=base_config.max_tokens,
        timeout=base_config.timeout,
    )
    provider = create_provider(provider_config)
    planner = PlannerAgent(provider=provider, config=provider_config)
    safety_guard = SafetyGuard()
    failure_detector = FailureDetector()
    reflection_engine = ReflectionEngine()

    start_time = time.perf_counter()
    plan = planner.create_plan(instruction)
    planning_time = time.perf_counter() - start_time

    safety_start = time.perf_counter()
    safety_result = safety_guard.validate_plan(plan)
    safety_time = time.perf_counter() - safety_start

    simulator = None
    simulator_status = "not initialized"
    try:
        env = MetaWorldEnvironment(task_name="pick-place-v3")
        env.initialize()
        simulator = env
        simulator_status = "initialized"
    except Exception as exc:  # pragma: no cover - defensive UI path
        simulator = None
        simulator_status = f"unavailable: {exc}"

    execution_logger = ExecutionLogger(str(experiment_dir))
    execution_engine = ExecutionEngine(simulator=simulator, logger=execution_logger)

    execution_start = time.perf_counter()
    execution_result = execution_engine.execute_plan(plan, safety_result=safety_result)
    execution_time = time.perf_counter() - execution_start

    failure_report = None
    reflection_report = None
    recovery_decision = None
    if not execution_result.success and execution_result.errors:
        failure_report = failure_detector.detect(execution_result, safety_result=safety_result)
        reflection_report = reflection_engine.analyze_failure(failure_report)
        recovery_decision = reflection_engine.recommend_recovery(failure_report)

    summary = (
        "Execution completed successfully and safely."
        if safety_result.approved and execution_result.success
        else "Execution completed with warnings or a rejected safety plan."
    )
    if failure_report is not None and reflection_report is not None:
        summary = f"Execution failed and self-correction recommended: {recovery_decision.strategy if recovery_decision else 'unknown'}"

    report = {
        "experiment_id": experiment_dir.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "instruction": instruction,
        "provider": provider_name,
        "planning_time": round(planning_time, 4),
        "safety_time": round(safety_time, 4),
        "execution_time": round(execution_time, 4),
        "total_time": round(planning_time + safety_time + execution_time, 4),
        "safety_status": "approved" if safety_result.approved else "rejected",
        "task_success": bool(execution_result.success),
        "simulator_status": simulator_status,
        "generated_plan": _serialize(plan.model_dump()),
        "safety_report": _serialize(safety_result.model_dump()),
        "execution_status": _serialize(execution_result.model_dump()),
        "failure_report": _serialize(failure_report.model_dump()) if failure_report else None,
        "reflection_report": _serialize(reflection_report.model_dump()) if reflection_report else None,
        "recovery_decision": _serialize(recovery_decision.model_dump()) if recovery_decision else None,
        "summary": summary,
    }

    (experiment_dir / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_csv_path = experiment_dir / "summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = __import__("csv").writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["planning_time", report["planning_time"]])
        writer.writerow(["safety_time", report["safety_time"]])
        writer.writerow(["execution_time", report["execution_time"]])
        writer.writerow(["total_time", report["total_time"]])
        writer.writerow(["task_success", str(report["task_success"]).lower()])

    history = _load_history()
    history.insert(0, report)
    _save_history(history[:50])
    return report


def _get_history_df(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["timestamp", "instruction", "provider", "planning_time", "execution_time", "total_time", "safety_status", "task_success"])
    frame = pd.DataFrame(history)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp", ascending=False)
    return frame


st.title("SafeLite Research Dashboard")
st.caption("Run end-to-end planner, safety, execution, and reflection experiments from a single interface.")

with st.sidebar:
    st.header("Experiment Controls")
    provider_name = st.selectbox("LLM Provider", ["mock", "groq", "openrouter"], index=0)
    api_key = st.text_input("API Key", value="", type="password", help="Optional; leave blank to use the existing environment or fall back to mock.")
    run_button = st.button("Run Experiment", use_container_width=True, type="primary")
    if st.button("Clear History", use_container_width=True):
        _save_history([])
        st.session_state.pop("current_result", None)
        st.rerun()

instruction = st.text_area(
    "Natural Language Instruction",
    value="Pick up the red cube.",
    height=140,
    placeholder="Example: Pick up the red cube.",
)

if run_button:
    if not instruction.strip():
        st.warning("Please enter an instruction before running the experiment.")
    else:
        with st.spinner("Running the SafeLite pipeline..."):
            result = _run_experiment(instruction, provider_name, api_key)
            st.session_state["current_result"] = result

current_result = st.session_state.get("current_result")
history = _load_history()

if current_result is None and history:
    current_result = history[0]

if current_result:
    render_metrics_panel(current_result)
    render_results_panel(current_result)
else:
    st.info("No experiment has been run yet. Enter an instruction and click Run Experiment.")

render_history_panel(history)
render_visualizations(_get_history_df(history))
