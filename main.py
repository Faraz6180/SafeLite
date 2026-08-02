"""Command-line entry point for the SafeLite execution pipeline."""

from __future__ import annotations

from pipeline.orchestrator import PipelineOrchestrator


def main() -> None:
    """Run the complete pipeline for a user-entered instruction."""
    instruction = input("Enter a natural language instruction: ").strip()
    orchestrator = PipelineOrchestrator()
    report = orchestrator.execute(instruction)

    print("\nGenerated Plan")
    print("--------------")
    print(report["generated_plan"])

    print("\nSafety Report")
    print("-------------")
    print(report["safety_report"])

    print("\nExecution Status")
    print("----------------")
    execution_status = report["execution_status"]
    print(f"Success: {execution_status['success']}")
    print(f"Completed: {len(execution_status['completed_actions'])}")
    print(f"Errors: {execution_status['errors']}")

    print("\nReflection Report")
    print("-----------------")
    if report.get("reflection_report"):
        print(report["reflection_report"])
    else:
        print("No reflection needed.")

    print("\nFinal Summary")
    print("------------")
    print(report["summary"])


if __name__ == "__main__":
    main()
