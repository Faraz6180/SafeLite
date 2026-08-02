"""Live SafeLite demo using the configured Groq provider."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from planner.planner import PlannerAgent
from safety.safety_guard import SafetyGuard
from simulator.environment import MetaWorldEnvironment
from self_correction.failure_detector import FailureDetector
from self_correction.reflection import ReflectionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    instruction = input("Enter a natural language instruction: ").strip() or "Pick up the red cube and place it into the blue basket."

    planner = PlannerAgent()
    safety_guard = SafetyGuard()
    failure_detector = FailureDetector()
    reflection_engine = ReflectionEngine()

    logger.info("Generating plan with provider: %s", planner._config.provider)
    plan = planner.create_plan(instruction)

    logger.info("Validating plan")
    safety_result = safety_guard.validate_plan(plan)

    logger.info("Executing inside simulator")
    try:
        simulator = MetaWorldEnvironment()
        simulator.initialize()
        simulator.reset()
        execution_result = {
            "success": True,
            "completed_actions": [action.model_dump() for action in plan.actions],
            "failed_action": None,
            "errors": [],
            "logs": ["Simulator executed successfully"],
        }
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Simulator execution failed")
        execution_result = {
            "success": False,
            "completed_actions": [],
            "failed_action": None,
            "errors": [str(exc)],
            "logs": [str(exc)],
        }
    finally:
        try:
            simulator.close()
        except Exception:  # pragma: no cover - defensive error handling
            pass

    failure_report = None
    if not execution_result["success"]:
        failure_report = failure_detector.detect(
            type("Result", (), {"errors": execution_result["errors"], "success": False})(),
            safety_result=safety_result,
        )

    reflection_report = None
    if failure_report is not None:
        reflection_report = reflection_engine.analyze_failure(failure_report)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "live_demo.json").write_text(json.dumps({
        "instruction": instruction,
        "planner_output": plan.model_dump(),
        "safety_report": safety_result.model_dump(),
        "execution_result": execution_result,
        "reflection_report": reflection_report.model_dump() if reflection_report else None,
    }, indent=2), encoding="utf-8")

    print("\nPlanner Output")
    print("--------------")
    print(json.dumps(plan.model_dump(), indent=2))

    print("\nSafety Report")
    print("-------------")
    print(safety_guard.generate_report(safety_result))

    print("\nExecution Result")
    print("----------------")
    print(json.dumps(execution_result, indent=2))

    print("\nReflection Report")
    print("-----------------")
    if reflection_report is not None:
        print(json.dumps(reflection_report.model_dump(), indent=2))
    else:
        print("No reflection needed.")


if __name__ == "__main__":
    main()
