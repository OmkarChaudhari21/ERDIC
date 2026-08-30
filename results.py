"""Loaders for stored evaluation and training results.

Pure functions from files and stores to plain data, kept free of Streamlit so they are
testable on their own. Every loader returns ``None`` for "nothing stored there" -- the
dashboards render that as an explicit empty state with the command that produces the
artifact, never as placeholder numbers -- and raises :class:`ResultsError` for a file
that exists but cannot be trusted, because a dashboard quietly skipping a corrupt report
would be indistinguishable from one that read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ResultsError(Exception):
    """A stored result exists but is unreadable or malformed."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsError(f"{path.name} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultsError(f"{path.name} does not contain a JSON object")
    return payload


# ---------------------------------------------------------------------------
# The RAG evaluation comparison (python -m evaluation.run)
# ---------------------------------------------------------------------------


def load_evaluation_comparison(path: Path) -> dict[str, Any] | None:
    """The comparison payload written by ``python -m evaluation.run``."""
    payload = _read_json(path)
    if payload is None:
        return None
    for key in ("context", "configurations", "deltas"):
        if key not in payload:
            raise ResultsError(f"{path.name} is missing the {key!r} section")
    return payload


def uses_fake_provider(comparison: dict[str, Any]) -> bool:
    """Whether the stored run was produced by the deterministic fake provider."""
    context = comparison.get("context", {})
    return "fake" in str(context.get("provider", "")) or "fake" in str(
        context.get("judge_provider", "")
    )


# ---------------------------------------------------------------------------
# Router training artifacts (scripts/train_router.py, scripts/evaluate_router.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """One stored training run: the manifest plus its measured metrics."""

    model: str
    backend: dict[str, Any]
    hyperparameters: dict[str, Any]
    parameters: dict[str, Any]
    dataset: dict[str, Any]
    duration_seconds: float
    #: ``{"validation": {loss, accuracy, macro_f1}, "test": {...}}`` -- real evaluations.
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The baseline-vs-fine-tuned router evaluation, when one was run.
    router_evaluation: dict[str, Any] | None = None


def load_training_run(artifacts_dir: Path) -> TrainingRun | None:
    """The training run stored under ``artifacts_dir``, or None when none exists."""
    manifest = _read_json(artifacts_dir / "training_manifest.json")
    if manifest is None:
        return None
    metrics = _read_json(artifacts_dir / "metrics.json") or {}
    evaluation = _read_json(artifacts_dir / "evaluation.json")
    return TrainingRun(
        model=str(manifest.get("model", "?")),
        backend=dict(manifest.get("backend", {})),
        hyperparameters=dict(manifest.get("hyperparameters", {})),
        parameters=dict(manifest.get("parameters", {})),
        dataset=dict(manifest.get("dataset", {})),
        duration_seconds=float(manifest.get("duration_seconds", 0.0)),
        metrics={
            split: {name: float(value) for name, value in values.items()}
            for split, values in metrics.items()
            if isinstance(values, dict)
        },
        router_evaluation=evaluation,
    )


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


def load_mlflow_runs(
    tracking_uri: str | None = None, *, limit: int = 100
) -> list[dict[str, Any]] | None:
    """Every run across every experiment in the store, newest first.

    Returns None when MLflow itself is not installed (the ``training`` extra), and an
    empty list when the store exists but holds nothing.
    """
    try:
        import mlflow
    except ImportError:
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        experiments = mlflow.search_experiments()
        runs: list[dict[str, Any]] = []
        for experiment in experiments:
            frame: Any = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                output_format="pandas",
            )
            for _, row in frame.iterrows():
                runs.append(
                    {
                        "experiment": experiment.name,
                        "run_name": row.get("tags.mlflow.runName", ""),
                        "run_id": row.get("run_id", ""),
                        "status": row.get("status", ""),
                        "started": str(row.get("start_time", ""))[:19],
                        "metrics": {
                            key.removeprefix("metrics."): value
                            for key, value in row.items()
                            if key.startswith("metrics.") and value == value  # not NaN
                        },
                        "params": {
                            key.removeprefix("params."): value
                            for key, value in row.items()
                            if key.startswith("params.") and value is not None
                        },
                    }
                )
    except Exception as exc:  # mlflow raises assorted store errors
        raise ResultsError(f"could not read the MLflow store: {exc}") from exc
    runs.sort(key=lambda run: str(run["started"]), reverse=True)
    return runs[:limit]
