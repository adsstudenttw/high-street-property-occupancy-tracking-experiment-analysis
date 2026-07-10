#!/usr/bin/env python3
"""Analyze MLflow tracking runs for object tracking experiments."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import textwrap
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

mlflow = None
np = None
pd = None
plt = None
sns = None
stats = None
ViewType = None
MlflowClient = None


# ==================================================
# CONFIG
# ==================================================

DEFAULT_EXPERIMENT_NAMES = ["MOTIP", "BoostTrack++", "SiamMOT"]
NORMALIZED_STAGES = [
    "baseline",
    "finetuning",
    "hyperparameter_tuning",
    "final_evaluation",
]
STAGE_ORDER = {stage: index for index, stage in enumerate(NORMALIZED_STAGES)}
ANALYSIS_METRICS = ["HOTA", "DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]
DECOMPOSITION_METRICS = ["DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_names": DEFAULT_EXPERIMENT_NAMES,
    "model_aliases": {
        "MOTIP": ["motip"],
        "BoostTrack++": ["boosttrack", "boosttrack++"],
        "SiamMOT": ["siammot", "siam mot"],
    },
    "model_detection_rules": {
        "MOTIP": [
            r"^motip_",
            r"\bmotip\b",
        ],
        "BoostTrack++": [
            r"^boosttrack_",
            r"\bboosttrack\b",
        ],
        "SiamMOT": [
            r"^siammot_",
            r"\bsiammot\b",
        ],
    },
    "stage_normalization_mapping": {
        "MOTIP": {
            "baseline_establishment": "baseline",
            "finetuning": "finetuning",
            "hyperparameter_tuning": "hyperparameter_tuning",
            "final_evaluation_from_tuning": "final_evaluation",
        },
        "BoostTrack++": {
            "baseline_eval": "baseline",
            "hpo_eval": "hyperparameter_tuning",
            "final_eval_best_hpo": "final_evaluation",
        },
        "SiamMOT": {
            "baseline_eval": "baseline",
            "fine_tune": "finetuning",
            "hpo": "hpo_parent_summary",
            "hpo_trial": "hyperparameter_tuning",
            "final_eval_best_hpo": "final_evaluation",
        },
    },
    "run_name_stage_rules": {
        "MOTIP": [
            (r"^motip_baseline_evaluation$", "baseline"),
            (r"^motip_finetuning$", "finetuning"),
            (r"^motip_hyperparameter_tuning_trial_\d+$", "hyperparameter_tuning"),
            (r"^motip_hyperparameter_tuning_final_evaluation$", "final_evaluation"),
        ],
        "BoostTrack++": [
            (r"^boosttrack_baseline_evaluation(_experiment)?$", "baseline"),
            (r"^boosttrack_hyperparameter_tuning_trial_\d+$", "hyperparameter_tuning"),
            (
                r"^boosttrack_hyperparameter_tuning_final_evaluation$",
                "final_evaluation",
            ),
            (r"^boosttrack_hyperparameter_tuning_experiment$", "hpo_parent_summary"),
        ],
        "SiamMOT": [
            (r"^siammot_baseline_evaluation$", "baseline"),
            (r"^siammot_finetuning$", "finetuning"),
            (r"^siammot_hyperparameter_tuning_trial_\d+$", "hyperparameter_tuning"),
            (r"^siammot_hyperparameter_tuning_experiment$", "hpo_parent_summary"),
            (r"^siammot_hyperparameter_tuning_final_evaluation$", "final_evaluation"),
        ],
    },
    "metric_field_preferences": {
        "MOTIP": {
            "baseline": ["HOTA"],
            "finetuning": ["HOTA", "epoch_HOTA"],
            "hyperparameter_tuning": ["HOTA", "epoch_HOTA"],
            "final_evaluation": ["HOTA"],
        },
        "BoostTrack++": {
            "baseline": ["val_hota", "best_val_hota"],
            "hyperparameter_tuning": ["val_hota"],
            "final_evaluation": ["test_hota"],
        },
        "SiamMOT": {
            "baseline": ["infer/mot/hota/hota", "infer/mot/hota"],
            "finetuning": ["val/mot/hota/hota", "val/mot/hota", "infer/mot/hota/hota"],
            "hyperparameter_tuning": [
                "val/mot/hota/hota",
                "val/mot/hota",
                "hpo/final_objective",
                "hpo/objective",
            ],
            "final_evaluation": ["infer/mot/hota/hota", "infer/mot/hota"],
        },
    },
    "normalized_metric_preferences": {
        "MOTIP": {
            "baseline": {
                "HOTA": ["HOTA"],
                "DetA": ["DetA"],
                "AssA": ["AssA"],
                "DetPr": ["DetPr"],
                "DetRe": ["DetRe"],
                "AssPr": ["AssPr"],
                "AssRe": ["AssRe"],
            },
            "finetuning": {
                "HOTA": ["epoch_HOTA", "HOTA"],
                "DetA": ["epoch_DetA", "DetA"],
                "AssA": ["epoch_AssA", "AssA"],
                "DetPr": ["epoch_DetPr", "DetPr"],
                "DetRe": ["epoch_DetRe", "DetRe"],
                "AssPr": ["epoch_AssPr", "AssPr"],
                "AssRe": ["epoch_AssRe", "AssRe"],
            },
            "hyperparameter_tuning": {
                "HOTA": ["epoch_HOTA", "HOTA"],
                "DetA": ["epoch_DetA", "DetA"],
                "AssA": ["epoch_AssA", "AssA"],
                "DetPr": ["epoch_DetPr", "DetPr"],
                "DetRe": ["epoch_DetRe", "DetRe"],
                "AssPr": ["epoch_AssPr", "AssPr"],
                "AssRe": ["epoch_AssRe", "AssRe"],
            },
            "final_evaluation": {
                "HOTA": ["HOTA"],
                "DetA": ["DetA"],
                "AssA": ["AssA"],
                "DetPr": ["DetPr"],
                "DetRe": ["DetRe"],
                "AssPr": ["AssPr"],
                "AssRe": ["AssRe"],
            },
        },
        "BoostTrack++": {
            "baseline": {
                "HOTA": ["val_hota", "best_val_hota"],
                "DetA": ["val_det_a"],
                "AssA": ["val_ass_a"],
                "DetPr": ["val_det_pr"],
                "DetRe": ["val_det_re"],
                "AssPr": ["val_ass_pr"],
                "AssRe": ["val_ass_re"],
            },
            "hyperparameter_tuning": {
                "HOTA": ["val_hota"],
                "DetA": ["val_det_a"],
                "AssA": ["val_ass_a"],
                "DetPr": ["val_det_pr"],
                "DetRe": ["val_det_re"],
                "AssPr": ["val_ass_pr"],
                "AssRe": ["val_ass_re"],
            },
            "final_evaluation": {
                "HOTA": ["test_hota"],
                "DetA": ["test_det_a"],
                "AssA": ["test_ass_a"],
                "DetPr": ["test_det_pr"],
                "DetRe": ["test_det_re"],
                "AssPr": ["test_ass_pr"],
                "AssRe": ["test_ass_re"],
            },
        },
        "SiamMOT": {
            "baseline": {
                "HOTA": ["infer/mot/hota/hota", "infer/mot/hota"],
                "DetA": ["infer/mot/hota/deta"],
                "AssA": ["infer/mot/hota/assa"],
                "DetPr": ["infer/mot/hota/detpr"],
                "DetRe": ["infer/mot/hota/detre"],
                "AssPr": ["infer/mot/hota/asspr"],
                "AssRe": ["infer/mot/hota/assre"],
            },
            "finetuning": {
                "HOTA": ["val/mot/hota/hota", "val/mot/hota", "infer/mot/hota/hota"],
                "DetA": ["val/mot/hota/deta"],
                "AssA": ["val/mot/hota/assa"],
                "DetPr": ["val/mot/hota/detpr"],
                "DetRe": ["val/mot/hota/detre"],
                "AssPr": ["val/mot/hota/asspr"],
                "AssRe": ["val/mot/hota/assre"],
            },
            "hyperparameter_tuning": {
                "HOTA": ["val/mot/hota/hota", "val/mot/hota", "hpo/final_objective", "hpo/objective"],
                "DetA": ["val/mot/hota/deta"],
                "AssA": ["val/mot/hota/assa"],
                "DetPr": ["val/mot/hota/detpr"],
                "DetRe": ["val/mot/hota/detre"],
                "AssPr": ["val/mot/hota/asspr"],
                "AssRe": ["val/mot/hota/assre"],
            },
            "final_evaluation": {
                "HOTA": ["infer/mot/hota/hota", "infer/mot/hota"],
                "DetA": ["infer/mot/hota/deta"],
                "AssA": ["infer/mot/hota/assa"],
                "DetPr": ["infer/mot/hota/detpr"],
                "DetRe": ["infer/mot/hota/detre"],
                "AssPr": ["infer/mot/hota/asspr"],
                "AssRe": ["infer/mot/hota/assre"],
            },
        },
    },
    "preferred_hyperparameters": {
        "MOTIP": [
            "LR",
            "WEIGHT_DECAY",
            "LR_BACKBONE_SCALE",
            "LR_DICTIONARY_SCALE",
            "LR_WARMUP_EPOCHS",
            "MAX_CLIP_NORM",
            "ID_LOSS_WEIGHT",
            "ASSIGNMENT_PROTOCOL",
            "DET_THRESH",
            "NEWBORN_THRESH",
            "ID_THRESH",
            "MISS_TOLERANCE",
            "AREA_THRESH",
        ],
        "BoostTrack++": [
            "det_thresh",
            "iou_threshold",
            "min_hits",
            "max_age",
            "lambda_iou",
            "lambda_mhd",
            "lambda_shape",
            "dlo_boost_coef",
            "use_dlo_boost",
            "use_duo_boost",
        ],
        "SiamMOT": [
            "BASE_LR",
            "WEIGHT_DECAY",
            "TRACK_THRESH",
            "START_TRACK_THRESH",
            "RESUME_TRACK_THRESH",
            "MAX_DORMANT_FRAMES",
            "TRACK_SCORE_THRESH",
            "MIN_TRACK_LENGTH",
        ],
    },
    "param_name_mapping": {
        "SiamMOT": {
            "optuna.solver_base_lr": "BASE_LR",
            "optuna.solver_weight_decay": "WEIGHT_DECAY",
            "optuna.model_track_thresh": "TRACK_THRESH",
            "optuna.model_start_track_thresh": "START_TRACK_THRESH",
            "optuna.model_resume_track_thresh": "RESUME_TRACK_THRESH",
            "optuna.model_max_dormant_frames": "MAX_DORMANT_FRAMES",
            "optuna.infer_track_score_thresh": "TRACK_SCORE_THRESH",
            "optuna.infer_min_track_length": "MIN_TRACK_LENGTH",
        }
    },
    "seed_candidates": [
        "seed",
        "random_seed",
        "trial_seed",
        "optuna.seed",
        "tags.seed",
        "tags.random_seed",
    ],
    "dataset_filter_defaults": [],
    "run_name_regex": {
        "hpo_trial_number": r"trial_(\d+)$",
    },
    "parent_child_handling_rules": {
        "BoostTrack++": {
            "exclude_parent_baseline_if_child_exists": True,
            "exclude_hpo_parent_summary": True,
        },
        "SiamMOT": {
            "exclude_hpo_parent_summary": True,
        },
    },
    "include_final_evaluation_in_core_comparisons": True,
    "include_only_finished_runs_in_primary_analysis": True,
}

SYSTEM_TAG_PREFIXES = ("mlflow.",)
SYSTEM_PARAM_PREFIXES = ("mlflow.",)
DATASET_FIELDS = [
    "dataset_key",
    "dataset",
    "benchmark",
    "train_split",
    "inference_split",
    "eval_split",
]
CORE_EXPORT_COLUMNS = [
    "run_id",
    "experiment_id",
    "experiment_name",
    "run_name",
    "status",
    "lifecycle_stage",
    "start_time",
    "end_time",
    "parent_run_id",
    "model",
    "raw_stage",
    "normalized_stage",
    "normalized_HOTA",
    "normalized_HOTA_source",
    "normalized_DetA",
    "normalized_DetA_source",
    "normalized_AssA",
    "normalized_AssA_source",
    "normalized_DetPr",
    "normalized_DetPr_source",
    "normalized_DetRe",
    "normalized_DetRe_source",
    "normalized_AssPr",
    "normalized_AssPr_source",
    "normalized_AssRe",
    "normalized_AssRe_source",
    "include_in_primary_analysis",
    "exclusion_reason",
    "is_incomplete_run",
    "dataset_key",
    "dataset",
    "benchmark",
    "train_split",
    "inference_split",
    "eval_split",
    "seed",
    "hpo_study_name",
    "hpo_trial_number",
    "hpo_stage_iter",
    "trial_state",
    "hpo_trial_state",
    "is_parent_run",
    "is_child_run",
    "run_role",
]


@dataclass
class PipelineArtifacts:
    raw_runs: pd.DataFrame
    cleaned_runs: pd.DataFrame
    descriptive_summary: pd.DataFrame
    improvement_summary: pd.DataFrame
    hota_question_answers: pd.DataFrame
    hyperparameter_question_answers: pd.DataFrame
    decomposition_summary: pd.DataFrame
    decomposition_improvement_summary: pd.DataFrame
    best_validation_to_final_decomposition: pd.DataFrame
    performance_profiles: pd.DataFrame
    decomposition_cross_model_comparison: pd.DataFrame
    decomposition_question_answers: pd.DataFrame
    variability_summary: pd.DataFrame
    hyperparameter_sensitivity: pd.DataFrame
    hpo_stage_summary: pd.DataFrame
    hpo_convergence_summary: pd.DataFrame
    hpo_top_trials: pd.DataFrame
    hpo_model_comparison: pd.DataFrame
    hpo_statistical_analysis: pd.DataFrame
    hpo_parameter_group_tests: pd.DataFrame
    hpo_cross_model_tests: pd.DataFrame
    cross_model_comparison: pd.DataFrame
    statistical_tests: pd.DataFrame
    warnings: list[str]


def load_dependencies() -> None:
    """Import third-party dependencies lazily so --help works before installation."""
    global mlflow, np, pd, plt, sns, stats, ViewType, MlflowClient

    import matplotlib.pyplot as plt_module
    import mlflow as mlflow_module
    import numpy as np_module
    import pandas as pd_module
    import seaborn as sns_module
    from mlflow.entities import ViewType as view_type_module
    from mlflow.tracking import MlflowClient as mlflow_client_module
    from scipy import stats as stats_module

    mlflow = mlflow_module
    np = np_module
    pd = pd_module
    plt = plt_module
    sns = sns_module
    stats = stats_module
    ViewType = view_type_module
    MlflowClient = mlflow_client_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MLflow tracking experiments for MOTIP, BoostTrack++, and SiamMOT."
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI. If omitted, MLflow environment defaults are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where CSVs, plots, and summaries will be written.",
    )
    parser.add_argument(
        "--experiment-names",
        nargs="+",
        default=None,
        help="Exact experiment names to analyze.",
    )
    parser.add_argument(
        "--dataset-filter",
        nargs="+",
        default=None,
        help="Optional dataset filter applied to dataset tags such as dataset_key or dataset.",
    )
    parser.add_argument(
        "--metric-override",
        action="append",
        default=[],
        metavar="MODEL:STAGE=FIELD1,FIELD2",
        help="Override metric field preferences for a model/stage pair.",
    )
    parser.add_argument(
        "--include-unfinished-in-primary-analysis",
        action="store_true",
        help="Include unfinished runs in the cleaned analysis dataset.",
    )
    parser.add_argument(
        "--exclude-final-evaluation-from-core-comparisons",
        action="store_true",
        help="Exclude final_evaluation runs from cleaned_runs.csv and stage summaries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if args.experiment_names:
        config["experiment_names"] = args.experiment_names
    if args.dataset_filter:
        config["dataset_filter_defaults"] = args.dataset_filter
    if args.include_unfinished_in_primary_analysis:
        config["include_only_finished_runs_in_primary_analysis"] = False
    if args.exclude_final_evaluation_from_core_comparisons:
        config["include_final_evaluation_in_core_comparisons"] = False
    apply_metric_overrides(config, args.metric_override)
    return config


def apply_metric_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        try:
            lhs, rhs = override.split("=", 1)
            model, stage = lhs.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --metric-override '{override}'. Use MODEL:STAGE=FIELD1,FIELD2."
            ) from exc
        fields = [field.strip() for field in rhs.split(",") if field.strip()]
        if not fields:
            raise ValueError(
                f"Invalid --metric-override '{override}': no fields provided."
            )
        config["metric_field_preferences"].setdefault(model, {})[stage] = fields


def ensure_output_dirs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def datetime_from_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()


def clean_scalar(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def maybe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def detect_model(
    experiment_name: str,
    run_name: str,
    tags: dict[str, str],
    config: dict[str, Any],
) -> str | None:
    if experiment_name in config["experiment_names"]:
        return experiment_name

    lower_candidates = [
        normalize_text(experiment_name).lower(),
        normalize_text(run_name).lower(),
        normalize_text(tags.get("model_name")).lower(),
    ]
    for model, aliases in config["model_aliases"].items():
        if any(
            alias.lower() in candidate
            for alias in aliases
            for candidate in lower_candidates
        ):
            return model

    for model, patterns in config["model_detection_rules"].items():
        if any(
            re.search(pattern, normalize_text(run_name), flags=re.IGNORECASE)
            for pattern in patterns
        ):
            return model

    return None


def normalize_stage(
    model: str | None,
    run_name: str,
    tags: dict[str, str],
    parent_run_id: str | None,
    config: dict[str, Any],
) -> tuple[str | None, str | None, str]:
    raw_stage = coalesce(tags.get("stage"), tags.get("latest_stage_name"))
    if model and raw_stage:
        mapped = config["stage_normalization_mapping"].get(model, {}).get(raw_stage)
        if mapped:
            return raw_stage, mapped, "tag"

    if model:
        for pattern, normalized in config["run_name_stage_rules"].get(model, []):
            if re.search(pattern, run_name, flags=re.IGNORECASE):
                return raw_stage, normalized, "run_name"

    if raw_stage and raw_stage in NORMALIZED_STAGES:
        return raw_stage, raw_stage, "tag_direct"

    if parent_run_id and raw_stage == "hpo":
        return raw_stage, "hpo_parent_summary", "tag"

    return raw_stage, None, "unknown"


def extract_seed(
    params: dict[str, str], tags: dict[str, str], config: dict[str, Any]
) -> Any:
    for candidate in config["seed_candidates"]:
        if candidate.startswith("tags."):
            value = tags.get(candidate.split(".", 1)[1])
        else:
            value = coalesce(params.get(candidate), tags.get(candidate))
        if value is not None:
            return value
    return None


def extract_dataset_fields(
    tags: dict[str, str], params: dict[str, str]
) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for field in DATASET_FIELDS:
        extracted[field] = coalesce(tags.get(field), params.get(field))
    return extracted


def extract_hpo_fields(
    run_name: str,
    params: dict[str, str],
    tags: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    trial_number = coalesce(
        tags.get("hpo_trial_number"), params.get("hpo_trial_number")
    )
    if trial_number is None:
        match = re.search(config["run_name_regex"]["hpo_trial_number"], run_name)
        if match:
            trial_number = match.group(1).lstrip("0") or "0"
    return {
        "hpo_study_name": coalesce(
            tags.get("hpo_study_name"), params.get("hpo_study_name")
        ),
        "hpo_trial_number": trial_number,
        "hpo_stage_iter": coalesce(
            tags.get("hpo_stage_iter"), tags.get("latest_stage_iter")
        ),
        "trial_state": coalesce(tags.get("trial_state"), params.get("trial_state")),
        "hpo_trial_state": coalesce(
            tags.get("hpo_trial_state"), params.get("hpo_trial_state")
        ),
    }


def extract_normalized_hyperparameters(
    model: str | None,
    params: dict[str, str],
    tags: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not model:
        return {}

    normalized: dict[str, Any] = {}
    source_params = dict(params)
    if model == "BoostTrack++":
        for key, value in parse_json_object(tags.get("fixed_params")).items():
            source_params.setdefault(str(key), value)

    mapped_param_names = config["param_name_mapping"].get(model, {})
    reverse_mapping = {
        normalized_name: raw_name
        for raw_name, normalized_name in mapped_param_names.items()
    }

    for hyperparameter in config["preferred_hyperparameters"].get(model, []):
        value = None
        if hyperparameter in source_params:
            value = source_params[hyperparameter]
        elif hyperparameter in reverse_mapping:
            value = source_params.get(reverse_mapping[hyperparameter])
        else:
            for raw_name, normalized_name in mapped_param_names.items():
                if normalized_name == hyperparameter and raw_name in source_params:
                    value = source_params[raw_name]
                    break
        normalized[f"hp_{hyperparameter}"] = value
    return normalized


def select_metric_source(
    model: str | None,
    normalized_stage: str | None,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> tuple[float | None, str | None]:
    if not model or not normalized_stage:
        return None, None
    candidate_fields = (
        config["metric_field_preferences"].get(model, {}).get(normalized_stage, [])
    )
    for field in candidate_fields:
        if field in metrics:
            return safe_float(metrics[field]), field
    return None, None


def select_normalized_metric_source(
    metric_name: str,
    model: str | None,
    normalized_stage: str | None,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> tuple[float | None, str | None]:
    if not model or not normalized_stage:
        return None, None
    candidate_fields = (
        config["normalized_metric_preferences"]
        .get(model, {})
        .get(normalized_stage, {})
        .get(metric_name, [])
    )
    if not candidate_fields and metric_name == "HOTA":
        return select_metric_source(model, normalized_stage, metrics, config)
    for field in candidate_fields:
        if field in metrics:
            return safe_float(metrics[field]), field
    return None, None


def select_all_normalized_metrics(
    model: str | None,
    normalized_stage: str | None,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> dict[str, tuple[float | None, str | None]]:
    return {
        metric_name: select_normalized_metric_source(
            metric_name, model, normalized_stage, metrics, config
        )
        for metric_name in ANALYSIS_METRICS
    }


def flatten_run(
    run: Any,
    experiment_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    tags = {key: clean_scalar(value) for key, value in run.data.tags.items()}
    params = {key: clean_scalar(value) for key, value in run.data.params.items()}
    metrics = {key: clean_scalar(value) for key, value in run.data.metrics.items()}

    run_name = coalesce(
        getattr(run.info, "run_name", None),
        tags.get("mlflow.runName"),
        run.info.run_id,
    )
    parent_run_id = tags.get("mlflow.parentRunId")
    model = detect_model(experiment_name, run_name, tags, config)
    raw_stage, normalized_stage, stage_source = normalize_stage(
        model=model,
        run_name=run_name,
        tags=tags,
        parent_run_id=parent_run_id,
        config=config,
    )
    normalized_metric_values = select_all_normalized_metrics(
        model, normalized_stage, metrics, config
    )
    normalized_hota, hota_source = normalized_metric_values["HOTA"]

    row: dict[str, Any] = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "status": run.info.status,
        "lifecycle_stage": run.info.lifecycle_stage,
        "start_time": datetime_from_ms(run.info.start_time),
        "end_time": datetime_from_ms(run.info.end_time),
        "parent_run_id": parent_run_id,
        "model": model,
        "raw_stage": raw_stage,
        "normalized_stage": normalized_stage,
        "stage_detection_source": stage_source,
        "normalized_HOTA": normalized_hota,
        "normalized_HOTA_source": hota_source,
        "seed": extract_seed(params, tags, config),
    }
    for metric_name in DECOMPOSITION_METRICS:
        metric_value, metric_source = normalized_metric_values[metric_name]
        row[f"normalized_{metric_name}"] = metric_value
        row[f"normalized_{metric_name}_source"] = metric_source
    row.update(extract_dataset_fields(tags, params))
    row.update(extract_hpo_fields(run_name, params, tags, config))
    row.update(extract_normalized_hyperparameters(model, params, tags, config))

    for key, value in tags.items():
        row[f"tag.{key}"] = value
    for key, value in params.items():
        row[f"param.{key}"] = value
    for key, value in metrics.items():
        row[f"metric.{key}"] = value
    return row


def fetch_target_experiments(
    client: MlflowClient,
    experiment_names: list[str],
    logger: logging.Logger,
) -> tuple[list[Any], list[str]]:
    warnings: list[str] = []
    experiments = client.search_experiments(view_type=ViewType.ALL)
    by_name: dict[str, list[Any]] = defaultdict(list)
    for experiment in experiments:
        by_name[experiment.name].append(experiment)

    selected: list[Any] = []
    for name in experiment_names:
        matches = by_name.get(name, [])
        active_matches = [exp for exp in matches if exp.lifecycle_stage == "active"]
        deleted_matches = [exp for exp in matches if exp.lifecycle_stage != "active"]
        if active_matches:
            selected.append(active_matches[0])
            if len(active_matches) > 1:
                warning = f"Multiple active experiments found for '{name}'. Using experiment_id={active_matches[0].experiment_id}."
                warnings.append(warning)
                logger.warning(warning)
        elif deleted_matches:
            warning = f"Expected experiment '{name}' exists only in deleted state; skipping it."
            warnings.append(warning)
            logger.warning(warning)
        else:
            warning = f"Expected experiment '{name}' was not found."
            warnings.append(warning)
            logger.warning(warning)
    return selected, warnings


def fetch_runs_for_experiment(
    client: MlflowClient,
    experiment_id: str,
    logger: logging.Logger,
) -> list[Any]:
    runs: list[Any] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    logger.info("Fetched %s active runs for experiment_id=%s", len(runs), experiment_id)
    return runs


def build_raw_dataframe(
    experiments: list[Any],
    client: MlflowClient,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for run in fetch_runs_for_experiment(client, experiment.experiment_id, logger):
            rows.append(flatten_run(run, experiment.name, config))

    if not rows:
        return pd.DataFrame(columns=CORE_EXPORT_COLUMNS)

    raw_df = pd.DataFrame(rows)
    raw_df = raw_df.drop_duplicates(subset=["run_id"]).reset_index(drop=True)
    return raw_df


def annotate_run_hierarchy(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raw_df["is_parent_run"] = pd.Series(dtype=bool)
        raw_df["is_child_run"] = pd.Series(dtype=bool)
        raw_df["run_role"] = pd.Series(dtype=str)
        return raw_df

    child_parent_ids = set(raw_df["parent_run_id"].dropna().astype(str))
    raw_df["is_child_run"] = raw_df["parent_run_id"].notna()
    raw_df["is_parent_run"] = raw_df["run_id"].astype(str).isin(child_parent_ids)
    raw_df["run_role"] = np.where(
        raw_df["is_parent_run"] & raw_df["is_child_run"],
        "parent_and_child",
        np.where(
            raw_df["is_parent_run"],
            "parent",
            np.where(raw_df["is_child_run"], "child", "standalone"),
        ),
    )
    raw_df.loc[raw_df["normalized_stage"] == "hpo_parent_summary", "run_role"] = (
        "summary_parent"
    )
    return raw_df


def dataset_matches(row: pd.Series, dataset_filters: list[str]) -> bool:
    if not dataset_filters:
        return True
    candidates = [normalize_text(row.get(field)).lower() for field in DATASET_FIELDS]
    return any(
        filter_value.lower() in candidate
        for filter_value in dataset_filters
        for candidate in candidates
    )


def is_siammot_pruned_hpo_trial(row: pd.Series) -> bool:
    """Treat explicitly pruned SiamMOT HPO trials as analysis-eligible."""
    model = row.get("model")
    stage = row.get("normalized_stage")
    status = normalize_text(row.get("status")).upper()
    hpo_trial_state = normalize_text(row.get("hpo_trial_state")).upper()
    trial_state = normalize_text(row.get("trial_state")).upper()
    return (
        model == "SiamMOT"
        and stage == "hyperparameter_tuning"
        and status == "KILLED"
        and (hpo_trial_state == "PRUNED" or trial_state == "PRUNED")
    )


def choose_analysis_eligibility(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    if raw_df.empty:
        raw_df["include_in_primary_analysis"] = pd.Series(dtype=bool)
        raw_df["exclusion_reason"] = pd.Series(dtype=str)
        raw_df["is_incomplete_run"] = pd.Series(dtype=bool)
        return raw_df

    reasons: list[list[str]] = []
    dataset_filters = config["dataset_filter_defaults"]

    boosttrack_baseline_child_exists = set(
        raw_df.loc[
            (raw_df["model"] == "BoostTrack++")
            & (raw_df["normalized_stage"] == "baseline")
            & (raw_df["is_child_run"]),
            "parent_run_id",
        ]
        .dropna()
        .astype(str)
    )

    for _, row in raw_df.iterrows():
        row_reasons: list[str] = []
        status = normalize_text(row.get("status")).upper()
        lifecycle_stage = normalize_text(row.get("lifecycle_stage")).lower()
        stage = row.get("normalized_stage")
        model = row.get("model")
        run_name = normalize_text(row.get("run_name"))
        is_pruned_siammot_trial = is_siammot_pruned_hpo_trial(row)

        if lifecycle_stage != "active":
            row_reasons.append("inactive_or_deleted_run")
        if model is None:
            row_reasons.append("unknown_model")
        if stage not in NORMALIZED_STAGES:
            if stage == "hpo_parent_summary":
                row_reasons.append("summary_parent_run")
            else:
                row_reasons.append("non_comparable_stage")
        if pd.isna(row.get("normalized_HOTA")):
            row_reasons.append("missing_normalized_hota")
        if (
            config["include_only_finished_runs_in_primary_analysis"]
            and status != "FINISHED"
            and not is_pruned_siammot_trial
        ):
            row_reasons.append("unfinished_run")
        if (
            model == "BoostTrack++"
            and stage == "baseline"
            and row.get("run_id") in boosttrack_baseline_child_exists
        ):
            row_reasons.append("boosttrack_parent_baseline_summary")
        if (
            model == "BoostTrack++"
            and stage == "baseline"
            and run_name.endswith("_experiment")
            and row.get("is_parent_run")
        ):
            row_reasons.append("boosttrack_parent_baseline_summary")
        if (
            model == "BoostTrack++"
            and run_name == "boosttrack_hyperparameter_tuning_experiment"
        ):
            row_reasons.append("boosttrack_hpo_parent_summary")
        if model == "SiamMOT" and stage == "hpo_parent_summary":
            row_reasons.append("siammot_hpo_parent_summary")
        if (
            stage == "final_evaluation"
            and not config["include_final_evaluation_in_core_comparisons"]
        ):
            row_reasons.append("final_evaluation_excluded_by_config")
        if not dataset_matches(row, dataset_filters):
            row_reasons.append("dataset_filter_mismatch")

        reasons.append(sorted(set(row_reasons)))

    raw_df = raw_df.copy()
    raw_df["exclusion_reason"] = [";".join(reason) for reason in reasons]
    raw_df["include_in_primary_analysis"] = raw_df["exclusion_reason"].eq("")
    raw_df["is_incomplete_run"] = raw_df.apply(
        lambda row: normalize_text(row.get("status")).upper() != "FINISHED"
        and not is_siammot_pruned_hpo_trial(row),
        axis=1,
    )

    excluded_count = int((~raw_df["include_in_primary_analysis"]).sum())
    logger.info("Marked %s runs as excluded from primary analysis", excluded_count)
    return raw_df


def prepare_cleaned_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw_df.loc[raw_df["include_in_primary_analysis"]].copy()
    if cleaned.empty:
        return cleaned

    rename_map = {"normalized_stage": "stage", "normalized_HOTA": "HOTA"}
    rename_map.update(
        {f"normalized_{metric_name}": metric_name for metric_name in DECOMPOSITION_METRICS}
    )
    cleaned = cleaned.rename(columns=rename_map)
    for metric_name in ANALYSIS_METRICS:
        if metric_name in cleaned.columns:
            cleaned[metric_name] = maybe_numeric(cleaned[metric_name])
    cleaned = cleaned.sort_values(
        ["model", "stage", "HOTA"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return cleaned


def confidence_interval(
    values: pd.Series, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    values = maybe_numeric(values).dropna()
    n = len(values)
    if n < 2:
        return None, None
    mean = float(values.mean())
    sem = float(stats.sem(values, nan_policy="omit"))
    if math.isnan(sem):
        return None, None
    interval = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    return float(interval[0]), float(interval[1])


def describe_group(group: pd.DataFrame) -> pd.Series:
    hota = maybe_numeric(group["HOTA"]).dropna()
    best_idx = hota.idxmax() if not hota.empty else None
    ci_low, ci_high = confidence_interval(hota)
    return pd.Series(
        {
            "n_runs": int(hota.count()),
            "mean_HOTA": float(hota.mean()) if not hota.empty else None,
            "median_HOTA": float(hota.median()) if not hota.empty else None,
            "std_HOTA": float(hota.std(ddof=1)) if len(hota) > 1 else None,
            "variance_HOTA": float(hota.var(ddof=1)) if len(hota) > 1 else None,
            "min_HOTA": float(hota.min()) if not hota.empty else None,
            "max_HOTA": float(hota.max()) if not hota.empty else None,
            "best_run_id": (
                group.loc[best_idx, "run_id"] if best_idx is not None else None
            ),
            "best_run_name": (
                group.loc[best_idx, "run_name"] if best_idx is not None else None
            ),
            "ci95_low_HOTA": ci_low,
            "ci95_high_HOTA": ci_high,
        }
    )


def build_descriptive_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    if cleaned_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "stage",
                "n_runs",
                "mean_HOTA",
                "median_HOTA",
                "std_HOTA",
                "variance_HOTA",
                "min_HOTA",
                "max_HOTA",
                "best_run_id",
                "best_run_name",
                "ci95_low_HOTA",
                "ci95_high_HOTA",
            ]
        )
    summary = (
        cleaned_df.groupby(["model", "stage"], dropna=False, sort=False)
        .apply(describe_group, include_groups=False)
        .reset_index()
    )
    summary["stage_order"] = summary["stage"].map(STAGE_ORDER)
    return (
        summary.sort_values(["model", "stage_order"])
        .drop(columns=["stage_order"])
        .reset_index(drop=True)
    )


def build_improvement_summary(
    cleaned_df: pd.DataFrame, descriptive_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cleaned_df.empty:
        return pd.DataFrame()

    stage_pairs = [
        ("baseline", "finetuning", "validation_to_validation"),
        ("finetuning", "hyperparameter_tuning", "validation_to_validation"),
        ("baseline", "hyperparameter_tuning", "validation_to_validation"),
        ("hyperparameter_tuning", "final_evaluation", "validation_to_test"),
        ("baseline", "final_evaluation", "validation_to_test"),
    ]

    for model in sorted(cleaned_df["model"].dropna().unique()):
        model_desc = descriptive_df.loc[descriptive_df["model"] == model].set_index(
            "stage"
        )
        for from_stage, to_stage, comparison_context in stage_pairs:
            if from_stage not in model_desc.index or to_stage not in model_desc.index:
                continue
            from_row = model_desc.loc[from_stage]
            to_row = model_desc.loc[to_stage]
            for statistic_label, from_value, to_value in [
                ("mean", from_row["mean_HOTA"], to_row["mean_HOTA"]),
                ("best", from_row["max_HOTA"], to_row["max_HOTA"]),
            ]:
                if pd.isna(from_value) or pd.isna(to_value):
                    continue
                absolute_improvement = float(to_value - from_value)
                relative_improvement_pct = None
                if from_value not in (0, None) and not pd.isna(from_value):
                    relative_improvement_pct = float(
                        (absolute_improvement / from_value) * 100.0
                    )
                rows.append(
                    {
                        "model": model,
                        "comparison_context": comparison_context,
                        "comparison_type": statistic_label,
                        "from_stage": from_stage,
                        "to_stage": to_stage,
                        "from_HOTA": float(from_value),
                        "to_HOTA": float(to_value),
                        "absolute_improvement": absolute_improvement,
                        "relative_improvement_pct": relative_improvement_pct,
                    }
                )
    return pd.DataFrame(rows)


def build_decomposition_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cleaned_df.empty:
        return pd.DataFrame()
    for (model, stage), group in cleaned_df.groupby(["model", "stage"], sort=False):
        best_idx = group["HOTA"].idxmax() if group["HOTA"].notna().any() else None
        best_row = group.loc[best_idx] if best_idx is not None else None
        row: dict[str, Any] = {
            "model": model,
            "stage": stage,
            "n_runs": int(len(group)),
            "representative_basis": "best_observed_stage_run",
            "best_run_id": best_row.get("run_id") if best_row is not None else None,
            "best_run_name": best_row.get("run_name") if best_row is not None else None,
        }
        for metric_name in ANALYSIS_METRICS:
            if metric_name not in group.columns:
                continue
            values = maybe_numeric(group[metric_name]).dropna()
            row[f"mean_{metric_name}"] = float(values.mean()) if not values.empty else None
            row[f"median_{metric_name}"] = (
                float(values.median()) if not values.empty else None
            )
            best_value = best_row.get(metric_name) if best_row is not None else None
            row[f"best_{metric_name}"] = (
                float(best_value) if best_value is not None and pd.notna(best_value) else None
            )
        if row.get("best_DetA") is not None and row.get("best_AssA") is not None:
            gap = float(row["best_DetA"] - row["best_AssA"])
            if abs(gap) < 1.0:
                row["dominant_component"] = "balanced"
            elif gap > 0:
                row["dominant_component"] = "detection_dominant"
            else:
                row["dominant_component"] = "association_dominant"
        else:
            row["dominant_component"] = None
        if row.get("mean_DetA") is not None and row.get("mean_AssA") is not None:
            mean_gap = float(row["mean_DetA"] - row["mean_AssA"])
            if abs(mean_gap) < 1.0:
                row["mean_dominant_component"] = "balanced"
            elif mean_gap > 0:
                row["mean_dominant_component"] = "detection_dominant"
            else:
                row["mean_dominant_component"] = "association_dominant"
        else:
            row["mean_dominant_component"] = None
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["stage_order"] = summary["stage"].map(STAGE_ORDER)
    return (
        summary.sort_values(["model", "stage_order"])
        .drop(columns=["stage_order"])
        .reset_index(drop=True)
    )


def classify_precision_recall_profile(
    precision: float | None,
    recall: float | None,
    tolerance: float = 1.0,
) -> str | None:
    if precision is None or recall is None or pd.isna(precision) or pd.isna(recall):
        return None
    gap = float(precision - recall)
    if abs(gap) <= tolerance:
        return "balanced"
    return "precision_oriented" if gap > 0 else "recall_oriented"


def build_decomposition_improvement_summary(
    decomposition_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decomposition_summary.empty:
        return pd.DataFrame()
    stage_pairs = [
        ("baseline", "finetuning", "validation_to_validation"),
        ("finetuning", "hyperparameter_tuning", "validation_to_validation"),
        ("baseline", "hyperparameter_tuning", "validation_to_validation"),
        ("hyperparameter_tuning", "final_evaluation", "validation_to_test"),
        ("baseline", "final_evaluation", "validation_to_test"),
    ]
    for model in sorted(decomposition_summary["model"].dropna().unique()):
        model_df = decomposition_summary.loc[
            decomposition_summary["model"] == model
        ].set_index("stage")
        for from_stage, to_stage, comparison_context in stage_pairs:
            if from_stage not in model_df.index or to_stage not in model_df.index:
                continue
            from_row = model_df.loc[from_stage]
            to_row = model_df.loc[to_stage]
            row: dict[str, Any] = {
                "model": model,
                "comparison_context": comparison_context,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "comparison_basis": "best_observed_stage_run",
            }
            for metric_name in ANALYSIS_METRICS:
                from_value = from_row.get(f"best_{metric_name}")
                to_value = to_row.get(f"best_{metric_name}")
                if pd.isna(from_value) or pd.isna(to_value):
                    row[f"delta_{metric_name}"] = None
                else:
                    row[f"delta_{metric_name}"] = float(to_value - from_value)
            deta_delta = row.get("delta_DetA")
            assa_delta = row.get("delta_AssA")
            if deta_delta is None or assa_delta is None:
                row["primary_driver"] = None
            elif abs(deta_delta - assa_delta) <= 1.0:
                row["primary_driver"] = "balanced"
            elif deta_delta > assa_delta:
                row["primary_driver"] = "detection_gain"
            else:
                row["primary_driver"] = "association_gain"
            row["detection_profile_change"] = classify_precision_recall_profile(
                row.get("delta_DetPr"), row.get("delta_DetRe"), tolerance=0.5
            )
            row["association_profile_change"] = classify_precision_recall_profile(
                row.get("delta_AssPr"), row.get("delta_AssRe"), tolerance=0.5
            )
            rows.append(row)
    return pd.DataFrame(rows)


def classify_component_transition(
    delta_hota: float | None,
    delta_deta: float | None,
    delta_assa: float | None,
) -> str | None:
    values = [delta_hota, delta_deta, delta_assa]
    if any(value is None or pd.isna(value) for value in values):
        return None
    delta_hota = float(delta_hota)
    delta_deta = float(delta_deta)
    delta_assa = float(delta_assa)
    if abs(delta_deta - delta_assa) <= 1.0:
        dominant = "balanced"
    elif delta_deta < delta_assa:
        dominant = "detection"
    else:
        dominant = "association"
    if delta_hota < 0:
        return f"{dominant}_driven_drop" if dominant != "balanced" else "balanced_drop"
    if delta_hota > 0:
        return (
            f"{dominant}_driven_improvement"
            if dominant != "balanced"
            else "balanced_improvement"
        )
    return f"{dominant}_change" if dominant != "balanced" else "balanced_change"


def build_best_validation_to_final_decomposition(
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cleaned_df.empty:
        return pd.DataFrame()

    for model in sorted(cleaned_df["model"].dropna().unique()):
        hpo_group = cleaned_df.loc[
            (cleaned_df["model"] == model)
            & (cleaned_df["stage"] == "hyperparameter_tuning")
        ].copy()
        final_group = cleaned_df.loc[
            (cleaned_df["model"] == model)
            & (cleaned_df["stage"] == "final_evaluation")
        ].copy()
        if hpo_group.empty or final_group.empty:
            continue

        best_hpo = hpo_group.sort_values(
            ["HOTA", "run_id"], ascending=[False, True]
        ).iloc[0]
        final_row = final_group.sort_values(
            ["HOTA", "run_id"], ascending=[False, True]
        ).iloc[0]

        row: dict[str, Any] = {
            "model": model,
            "best_validation_run_id": best_hpo.get("run_id"),
            "best_validation_run_name": best_hpo.get("run_name"),
            "final_evaluation_run_id": final_row.get("run_id"),
            "final_evaluation_run_name": final_row.get("run_name"),
        }
        for metric_name in ANALYSIS_METRICS:
            best_value = best_hpo.get(metric_name)
            final_value = final_row.get(metric_name)
            row[f"best_validation_{metric_name}"] = (
                float(best_value) if pd.notna(best_value) else None
            )
            row[f"final_evaluation_{metric_name}"] = (
                float(final_value) if pd.notna(final_value) else None
            )
            if pd.notna(best_value) and pd.notna(final_value):
                row[f"delta_{metric_name}"] = float(final_value - best_value)
            else:
                row[f"delta_{metric_name}"] = None

        row["dominant_transition_driver"] = classify_component_transition(
            row.get("delta_HOTA"),
            row.get("delta_DetA"),
            row.get("delta_AssA"),
        )
        row["detection_profile_change"] = classify_precision_recall_profile(
            row.get("delta_DetPr"), row.get("delta_DetRe"), tolerance=0.5
        )
        row["association_profile_change"] = classify_precision_recall_profile(
            row.get("delta_AssPr"), row.get("delta_AssRe"), tolerance=0.5
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def select_profile_reference_stage(model_df: pd.DataFrame) -> str | None:
    for stage in ["final_evaluation", "hyperparameter_tuning", "finetuning", "baseline"]:
        if stage in model_df.index:
            return stage
    return None


def build_performance_profiles(
    decomposition_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decomposition_summary.empty:
        return pd.DataFrame()
    for model in sorted(decomposition_summary["model"].dropna().unique()):
        model_df = decomposition_summary.loc[
            decomposition_summary["model"] == model
        ].set_index("stage")
        reference_stage = select_profile_reference_stage(model_df)
        if reference_stage is None:
            continue
        row = model_df.loc[reference_stage]
        deta = row.get("best_DetA")
        assa = row.get("best_AssA")
        det_profile = classify_precision_recall_profile(
            row.get("best_DetPr"), row.get("best_DetRe")
        )
        ass_profile = classify_precision_recall_profile(
            row.get("best_AssPr"), row.get("best_AssRe")
        )
        if deta is None or assa is None or pd.isna(deta) or pd.isna(assa):
            dominant_profile = None
        elif abs(float(deta) - float(assa)) <= 1.0:
            dominant_profile = "balanced"
        elif deta > assa:
            dominant_profile = "detection_dominant"
        else:
            dominant_profile = "association_dominant"
        rows.append(
            {
                "model": model,
                "reference_stage": reference_stage,
                "performance_profile": dominant_profile,
                "detection_profile": det_profile,
                "association_profile": ass_profile,
                "profile_basis": "best_observed_stage_run",
                "best_DetA": deta,
                "best_AssA": assa,
                "best_DetPr": row.get("best_DetPr"),
                "best_DetRe": row.get("best_DetRe"),
                "best_AssPr": row.get("best_AssPr"),
                "best_AssRe": row.get("best_AssRe"),
            }
        )
    return pd.DataFrame(rows)


def build_decomposition_cross_model_comparison(
    decomposition_summary: pd.DataFrame,
) -> pd.DataFrame:
    if decomposition_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for stage in [stage for stage in NORMALIZED_STAGES if stage in decomposition_summary["stage"].unique()]:
        stage_df = decomposition_summary.loc[decomposition_summary["stage"] == stage].copy()
        if stage_df.empty:
            continue
        max_deta = stage_df["best_DetA"].max() if "best_DetA" in stage_df else None
        max_assa = stage_df["best_AssA"].max() if "best_AssA" in stage_df else None
        for _, row in stage_df.iterrows():
            rows.append(
                {
                    "stage": stage,
                    "model": row["model"],
                    "comparison_basis": "best_observed_stage_run",
                    "best_DetA": row.get("best_DetA"),
                    "best_AssA": row.get("best_AssA"),
                    "best_DetPr": row.get("best_DetPr"),
                    "best_DetRe": row.get("best_DetRe"),
                    "best_AssPr": row.get("best_AssPr"),
                    "best_AssRe": row.get("best_AssRe"),
                    "detection_minus_association_gap": (
                        float(row["best_DetA"] - row["best_AssA"])
                        if pd.notna(row.get("best_DetA")) and pd.notna(row.get("best_AssA"))
                        else None
                    ),
                    "detection_precision_recall_gap": (
                        float(row["best_DetPr"] - row["best_DetRe"])
                        if pd.notna(row.get("best_DetPr")) and pd.notna(row.get("best_DetRe"))
                        else None
                    ),
                    "association_precision_recall_gap": (
                        float(row["best_AssPr"] - row["best_AssRe"])
                        if pd.notna(row.get("best_AssPr")) and pd.notna(row.get("best_AssRe"))
                        else None
                    ),
                    "is_best_detection_model": (
                        pd.notna(row.get("best_DetA")) and row.get("best_DetA") == max_deta
                    ),
                    "is_best_association_model": (
                        pd.notna(row.get("best_AssA")) and row.get("best_AssA") == max_assa
                    ),
                }
            )
    return pd.DataFrame(rows)


def classify_breadth_of_change(
    deta_delta: float | None,
    assa_delta: float | None,
) -> str | None:
    if deta_delta is None or assa_delta is None:
        return None
    if pd.isna(deta_delta) or pd.isna(assa_delta):
        return None
    if deta_delta > 0 and assa_delta > 0:
        return "broad_gain"
    if deta_delta < 0 and assa_delta < 0:
        return "broad_decline"
    return "narrow_or_mixed_change"


def build_decomposition_question_answers(
    cross_model_df: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    decomposition_improvement_summary: pd.DataFrame,
    best_validation_to_final_decomposition: pd.DataFrame,
    performance_profiles: pd.DataFrame,
    decomposition_cross_model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in performance_profiles.iterrows():
        rows.append(
            {
                "question_id": "Q1",
                "question": "Is the model mainly limited by detection quality or association quality?",
                "scope": row["model"],
                "answer": (
                    f"At {row['reference_stage']}, {row['model']} was {row['performance_profile']} "
                    f"(DetA={format_optional(row['best_DetA'])}, AssA={format_optional(row['best_AssA'])})."
                ),
                "supporting_tables": "decomposition_summary.csv;performance_profiles.csv",
                "supporting_plots": "deta_vs_assa_by_model_stage.png",
            }
        )
        rows.append(
            {
                "question_id": "Q4",
                "question": "Is the model precision-oriented or recall-oriented in detection?",
                "scope": row["model"],
                "answer": (
                    f"At {row['reference_stage']}, {row['model']} had a {row['detection_profile']} detection profile "
                    f"(DetPr={format_optional(row['best_DetPr'])}, DetRe={format_optional(row['best_DetRe'])})."
                ),
                "supporting_tables": "decomposition_summary.csv;performance_profiles.csv;decomposition_cross_model_comparison.csv",
                "supporting_plots": "detection_precision_recall_profiles.png",
            }
        )
        rows.append(
            {
                "question_id": "Q5",
                "question": "Is the model precision-oriented or recall-oriented in association?",
                "scope": row["model"],
                "answer": (
                    f"At {row['reference_stage']}, {row['model']} had a {row['association_profile']} association profile "
                    f"(AssPr={format_optional(row['best_AssPr'])}, AssRe={format_optional(row['best_AssRe'])})."
                ),
                "supporting_tables": "decomposition_summary.csv;performance_profiles.csv;decomposition_cross_model_comparison.csv",
                "supporting_plots": "association_precision_recall_profiles.png",
            }
        )
        rows.append(
            {
                "question_id": "Q5B",
                "question": "What behavioral tracking profile does each model have?",
                "scope": row["model"],
                "answer": (
                    f"{row['model']} can be characterized as {row['performance_profile']} with "
                    f"{row['detection_profile']} detection behavior and {row['association_profile']} association behavior."
                ),
                "supporting_tables": "performance_profiles.csv",
                "supporting_plots": "deta_vs_assa_by_model_stage.png;detection_precision_recall_profiles.png;association_precision_recall_profiles.png",
            }
        )

    for _, row in decomposition_improvement_summary.iterrows():
        breadth = classify_breadth_of_change(row.get("delta_DetA"), row.get("delta_AssA"))
        if row["from_stage"] == "baseline" and row["to_stage"] == "finetuning":
            rows.append(
                {
                    "question_id": "Q2",
                    "question": "Did finetuning improve detection more, or association more?",
                    "scope": row["model"],
                    "answer": (
                        f"From baseline to finetuning, {row['model']} was driven mainly by {row['primary_driver']} "
                        f"(DetA delta={format_optional(row.get('delta_DetA'))}, AssA delta={format_optional(row.get('delta_AssA'))}; {breadth})."
                    ),
                    "supporting_tables": "decomposition_improvement_summary.csv",
                    "supporting_plots": "decomposition_improvements_across_stages.png",
                }
            )
        if row["to_stage"] == "hyperparameter_tuning":
            rows.append(
                {
                    "question_id": "Q3",
                    "question": "Did HPO improve detection more, or association more?",
                    "scope": row["model"],
                    "answer": (
                        f"From {row['from_stage']} into hyperparameter tuning, {row['model']} was driven mainly by {row['primary_driver']} "
                        f"(DetA delta={format_optional(row.get('delta_DetA'))}, AssA delta={format_optional(row.get('delta_AssA'))}; {breadth})."
                    ),
                    "supporting_tables": "decomposition_improvement_summary.csv",
                    "supporting_plots": "decomposition_improvements_across_stages.png",
                }
            )
        rows.append(
            {
                "question_id": "Q9",
                "question": "Are improvements broad and robust, or narrow and selective?",
                "scope": f"{row['model']} {row['from_stage']}->{row['to_stage']}",
                "answer": (
                    f"For {row['model']} from {row['from_stage']} to {row['to_stage']}, the change was {breadth}, "
                    f"with DetA delta={format_optional(row.get('delta_DetA'))} and AssA delta={format_optional(row.get('delta_AssA'))}."
                ),
                "supporting_tables": "decomposition_improvement_summary.csv",
                "supporting_plots": "decomposition_improvements_across_stages.png;detection_precision_recall_profiles.png;association_precision_recall_profiles.png",
            }
        )

    for _, row in best_validation_to_final_decomposition.iterrows():
        rows.append(
            {
                "question_id": "Q7",
                "question": "Why does final test performance differ from best validation performance during hyperparameter tuning?",
                "scope": row["model"],
                "answer": (
                    f"From the best HPO validation trial to final test, {row['model']} showed a {row['dominant_transition_driver']} "
                    f"(HOTA delta={format_optional(row.get('delta_HOTA'))}, DetA delta={format_optional(row.get('delta_DetA'))}, "
                    f"AssA delta={format_optional(row.get('delta_AssA'))}, detection profile change={row.get('detection_profile_change')}, "
                    f"association profile change={row.get('association_profile_change')})."
                ),
                "supporting_tables": "best_validation_to_final_decomposition.csv;improvement_summary.csv",
                "supporting_plots": "validation_best_vs_final_test.png;best_validation_to_final_decomposition.png",
            }
        )

    final_stage = decomposition_cross_model_comparison.loc[
        decomposition_cross_model_comparison["stage"] == "final_evaluation"
    ]
    if not final_stage.empty:
        best_det = final_stage.loc[final_stage["is_best_detection_model"]]
        best_ass = final_stage.loc[final_stage["is_best_association_model"]]
        if not best_det.empty or not best_ass.empty:
            det_text = (
                f"{best_det.iloc[0]['model']} (DetA={format_optional(best_det.iloc[0]['best_DetA'])})"
                if not best_det.empty
                else "n/a"
            )
            ass_text = (
                f"{best_ass.iloc[0]['model']} (AssA={format_optional(best_ass.iloc[0]['best_AssA'])})"
                if not best_ass.empty
                else "n/a"
            )
            rows.append(
                {
                    "question_id": "Q8",
                    "question": "Which model is strongest at detection, and which is strongest at association?",
                    "scope": "final_evaluation",
                    "answer": f"At final evaluation, the strongest detector was {det_text}; the strongest associator was {ass_text}.",
                    "supporting_tables": "decomposition_cross_model_comparison.csv",
                    "supporting_plots": "deta_vs_assa_by_model_stage.png",
                }
            )

    if not final_stage.empty and len(final_stage) >= 2:
        sortable = final_stage.dropna(subset=["best_DetA", "best_AssA"]).copy()
        if len(sortable) >= 2:
            sorted_by_hota = cross_model_df.dropna(subset=["final_evaluation_best_HOTA"]).sort_values(
                "final_evaluation_best_HOTA", ascending=False
            )
            if len(sorted_by_hota) >= 2:
                model_a = sorted_by_hota.iloc[0]["model"]
                model_b = sorted_by_hota.iloc[1]["model"]
                a_row = sortable.loc[sortable["model"] == model_a]
                b_row = sortable.loc[sortable["model"] == model_b]
                if not a_row.empty and not b_row.empty:
                    rows.append(
                        {
                            "question_id": "Q6",
                            "question": "Why do two models with similar HOTA differ in practice?",
                            "scope": f"{model_a} vs {model_b}",
                            "answer": (
                                f"{model_a} and {model_b} differ because {model_a} combines DetA={format_optional(a_row.iloc[0]['best_DetA'])} "
                                f"and AssA={format_optional(a_row.iloc[0]['best_AssA'])}, while {model_b} combines "
                                f"DetA={format_optional(b_row.iloc[0]['best_DetA'])} and AssA={format_optional(b_row.iloc[0]['best_AssA'])}."
                            ),
                            "supporting_tables": "cross_model_comparison.csv;decomposition_cross_model_comparison.csv;performance_profiles.csv",
                            "supporting_plots": "deta_vs_assa_by_model_stage.png;detection_precision_recall_profiles.png;association_precision_recall_profiles.png",
                        }
                    )

    if not cross_model_df.empty:
        best_final = cross_model_df.loc[cross_model_df["is_best_final_model"]]
        if not best_final.empty:
            model = best_final.iloc[0]["model"]
            profile = performance_profiles.loc[performance_profiles["model"] == model]
            profile_text = (
                f"{profile.iloc[0]['performance_profile']} with {profile.iloc[0]['detection_profile']} detection and {profile.iloc[0]['association_profile']} association behavior"
                if not profile.empty
                else "profile unavailable"
            )
            rows.append(
                {
                    "question_id": "Q10",
                    "question": "Which model is best overall, and why?",
                    "scope": model,
                    "answer": f"{model} was best overall on final HOTA, with a decomposition profile described as {profile_text}.",
                    "supporting_tables": "cross_model_comparison.csv;performance_profiles.csv;decomposition_cross_model_comparison.csv",
                    "supporting_plots": "deta_vs_assa_by_model_stage.png;validation_best_vs_final_test.png",
                }
            )

    return pd.DataFrame(rows)


def classify_hota_margin(margin: float | None) -> str | None:
    if margin is None or pd.isna(margin):
        return None
    margin = float(margin)
    if margin < 1.0:
        return "small"
    if margin < 5.0:
        return "moderate"
    return "large"


def build_hota_question_answers(
    descriptive_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    hpo_stage_summary: pd.DataFrame,
    hpo_convergence_summary: pd.DataFrame,
    hpo_model_comparison: pd.DataFrame,
    cross_model_df: pd.DataFrame,
    statistical_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not cross_model_df.empty:
        stage_columns = [
            ("baseline", "baseline_best_HOTA"),
            ("finetuning", "finetuning_best_HOTA"),
            ("hyperparameter_tuning", "hyperparameter_tuning_best_HOTA"),
            ("final_evaluation", "final_evaluation_best_HOTA"),
        ]
        stage_leaders: list[str] = []
        for stage, column in stage_columns:
            stage_rows = cross_model_df.dropna(subset=[column])
            if stage_rows.empty:
                continue
            best_row = stage_rows.loc[stage_rows[column].idxmax()]
            stage_leaders.append(
                f"{stage}={best_row['model']} ({format_optional(best_row[column])})"
            )
        best_final = cross_model_df.loc[cross_model_df["is_best_final_model"]]
        if not best_final.empty:
            best_final_row = best_final.iloc[0]
            rows.append(
                {
                    "question_id": "Q1",
                    "question": "Which model performs best overall?",
                    "scope": "overall",
                    "answer": (
                        f"{best_final_row['model']} performed best overall on final evaluation with "
                        f"HOTA={format_optional(best_final_row['final_evaluation_best_HOTA'])}. "
                        f"Stage leaders were {', '.join(stage_leaders)}."
                    ),
                    "supporting_tables": "cross_model_comparison.csv;descriptive_summary.csv",
                    "supporting_plots": "best_hota_by_stage.png;validation_best_vs_final_test.png",
                }
            )

    if not descriptive_df.empty:
        for model in sorted(descriptive_df["model"].dropna().unique()):
            model_rows = descriptive_df.loc[descriptive_df["model"] == model].copy()
            model_rows["stage_order"] = model_rows["stage"].map(STAGE_ORDER)
            model_rows = model_rows.sort_values("stage_order")
            stage_bits: list[str] = []
            for _, row in model_rows.iterrows():
                stage_bits.append(
                    f"{row['stage']}: mean={format_optional(row.get('mean_HOTA'))}, best={format_optional(row.get('max_HOTA'))}"
                )
            rows.append(
                {
                    "question_id": "Q2",
                    "question": "How good is each model at each experiment stage?",
                    "scope": model,
                    "answer": "; ".join(stage_bits) + ".",
                    "supporting_tables": "descriptive_summary.csv;cleaned_runs.csv",
                    "supporting_plots": "best_hota_by_stage.png;hota_distributions_by_model_stage.png",
                }
            )

    if not improvement_df.empty:
        best_improvement_df = improvement_df.loc[
            improvement_df["comparison_type"] == "best"
        ].copy()
        if not best_improvement_df.empty:
            for model in sorted(best_improvement_df["model"].dropna().unique()):
                model_rows = best_improvement_df.loc[
                    best_improvement_df["model"] == model
                ].copy()
                if model_rows.empty:
                    continue
                stage_bits = []
                for _, row in model_rows.iterrows():
                    stage_bits.append(
                        f"{row['from_stage']}->{row['to_stage']} {format_optional(row['absolute_improvement'])}"
                    )
                rows.append(
                    {
                        "question_id": "Q3",
                        "question": "How much does each stage improve or degrade performance?",
                        "scope": model,
                        "answer": (
                            f"Best-run HOTA changes for {model} were " + "; ".join(stage_bits) + "."
                        ),
                        "supporting_tables": "improvement_summary.csv",
                        "supporting_plots": "hota_improvement_across_stages.png;hota_improvement_across_stages_table.png",
                    }
                )

    if not hpo_stage_summary.empty:
        hpo_variability = variability_df.loc[
            variability_df["stage"] == "hyperparameter_tuning"
        ].set_index("model")
        for _, row in hpo_stage_summary.sort_values("model").iterrows():
            var_row = hpo_variability.loc[row["model"]] if row["model"] in hpo_variability.index else None
            stability_comment = (
                var_row.get("stability_comment")
                if isinstance(var_row, pd.Series)
                else None
            )
            rows.append(
                {
                    "question_id": "Q4",
                    "question": "How stable or variable is model performance across HPO trials?",
                    "scope": row["model"],
                    "answer": (
                        f"{row['model']} showed HPO variability with std={format_optional(row.get('std_HOTA'))}, "
                        f"IQR={format_optional(row.get('iqr_HOTA'))}, and CV={format_optional(var_row.get('coefficient_of_variation') if isinstance(var_row, pd.Series) else None)}"
                        + (f" ({stability_comment})." if stability_comment else ".")
                    ),
                    "supporting_tables": "variability_summary.csv;hpo_stage_summary.csv",
                    "supporting_plots": "hota_distributions_by_model_stage.png;hpo_hota_ecdf_by_model.png;hpo_stage_summary_overview.png",
                }
            )
            rows.append(
                {
                    "question_id": "Q5",
                    "question": "What is the gap between average performance and best-case performance?",
                    "scope": row["model"],
                    "answer": (
                        f"For {row['model']}, HPO mean={format_optional(row.get('mean_HOTA'))}, "
                        f"median={format_optional(row.get('median_HOTA'))}, best={format_optional(row.get('best_HOTA'))}, "
                        f"best-minus-median={format_optional(row.get('best_minus_median_HOTA'))}, and "
                        f"{format_optional((row.get('proportion_within_10pct_of_best') or 0) * 100 if pd.notna(row.get('proportion_within_10pct_of_best')) else None)}% "
                        f"of trials were within 10% of the best."
                    ),
                    "supporting_tables": "descriptive_summary.csv;hpo_stage_summary.csv;hpo_top_trials.csv",
                    "supporting_plots": "hpo_stage_summary_overview.png;hpo_hota_ecdf_by_model.png",
                }
            )

    if not hpo_convergence_summary.empty:
        for _, row in hpo_convergence_summary.sort_values("model").iterrows():
            rows.append(
                {
                    "question_id": "Q6",
                    "question": "How efficiently does each model benefit from hyperparameter tuning?",
                    "scope": row["model"],
                    "answer": (
                        f"{row['model']} reached 95% of its best HOTA by trial {int(row['first_trial_reaching_95pct_of_best'])}, "
                        f"found its best trial at order {int(row['trial_order_of_best'])}, and {row['convergence_comment'].lower()}"
                    ),
                    "supporting_tables": "hpo_convergence_summary.csv;hpo_model_comparison.csv",
                    "supporting_plots": "hpo_trial_progression_by_model.png;hpo_convergence_summary_overview.png",
                }
            )

    if not cross_model_df.empty:
        for _, row in cross_model_df.sort_values("model").iterrows():
            rows.append(
                {
                    "question_id": "Q7",
                    "question": "How much better is the tuned model than the baseline?",
                    "scope": row["model"],
                    "answer": (
                        f"For {row['model']}, baseline-to-best-validation gain was "
                        f"{format_optional(row.get('baseline_to_tuned_validation_gain'))} HOTA and "
                        f"baseline-to-final-test gain was {format_optional(row.get('baseline_to_final_test_gain'))} HOTA."
                    ),
                    "supporting_tables": "improvement_summary.csv;cross_model_comparison.csv",
                    "supporting_plots": "hota_improvement_across_stages.png;hpo_model_comparison_overview.png",
                }
            )

    if not improvement_df.empty:
        validation_transfer = improvement_df.loc[
            (improvement_df["comparison_type"] == "best")
            & (improvement_df["from_stage"] == "hyperparameter_tuning")
            & (improvement_df["to_stage"] == "final_evaluation")
        ].copy()
        for _, row in validation_transfer.sort_values("model").iterrows():
            relative = row.get("relative_improvement_pct")
            rows.append(
                {
                    "question_id": "Q8",
                    "question": "How well does best hyperparameter tuning trial validation HOTA transfer to final test HOTA?",
                    "scope": row["model"],
                    "answer": (
                        f"For {row['model']}, best validation HOTA moved from {format_optional(row.get('from_HOTA'))} "
                        f"to final test HOTA {format_optional(row.get('to_HOTA'))}, a change of "
                        f"{format_optional(row.get('absolute_improvement'))} HOTA"
                        + (
                            f" ({format_optional(relative)}% relative)."
                            if relative is not None and not pd.isna(relative)
                            else "."
                        )
                    ),
                    "supporting_tables": "improvement_summary.csv;cross_model_comparison.csv;hpo_model_comparison.csv",
                    "supporting_plots": "validation_best_vs_final_test.png;hota_improvement_across_stages.png",
                }
            )

    if not descriptive_df.empty:
        stage_best_df = descriptive_df.loc[
            :, ["model", "stage", "max_HOTA"]
        ].dropna(subset=["max_HOTA"])
        for stage in sorted(stage_best_df["stage"].unique(), key=lambda item: STAGE_ORDER.get(item, 99)):
            stage_rows = stage_best_df.loc[stage_best_df["stage"] == stage].sort_values(
                "max_HOTA", ascending=False
            )
            if len(stage_rows) < 2:
                continue
            best_row = stage_rows.iloc[0]
            second_row = stage_rows.iloc[1]
            margin = float(best_row["max_HOTA"] - second_row["max_HOTA"])
            margin_size = classify_hota_margin(margin)
            rows.append(
                {
                    "question_id": "Q9",
                    "question": "Are differences between models practically meaningful?",
                    "scope": stage,
                    "answer": (
                        f"At {stage}, {best_row['model']} led with HOTA={format_optional(best_row['max_HOTA'])}, "
                        f"ahead of {second_row['model']} by {format_optional(margin)} HOTA, which is a {margin_size} margin."
                    ),
                    "supporting_tables": "cross_model_comparison.csv;descriptive_summary.csv;statistical_tests.csv",
                    "supporting_plots": "best_hota_by_stage.png;validation_best_vs_final_test.png",
                }
            )

    if not cross_model_df.empty:
        best_final = cross_model_df.loc[cross_model_df["is_best_final_model"]]
        most_stable = cross_model_df.loc[cross_model_df["is_most_stable_model"]]
        validation_transfer = pd.DataFrame()
        if not improvement_df.empty:
            validation_transfer = improvement_df.loc[
                (improvement_df["comparison_type"] == "best")
                & (improvement_df["from_stage"] == "hyperparameter_tuning")
                & (improvement_df["to_stage"] == "final_evaluation")
            ].copy()
            if not validation_transfer.empty:
                validation_transfer["abs_drop"] = validation_transfer["absolute_improvement"].abs()
                validation_transfer = validation_transfer.sort_values("abs_drop")
        if not best_final.empty:
            best_model = best_final.iloc[0]["model"]
            stable_text = most_stable.iloc[0]["model"] if not most_stable.empty else "n/a"
            generalizer_text = (
                validation_transfer.iloc[0]["model"] if not validation_transfer.empty else "n/a"
            )
            synthesis = (
                f"{best_model} offered the strongest overall HOTA outcome."
                if best_model == stable_text == generalizer_text
                else f"The trade-off was split: best final HOTA={best_model}, most stable HPO={stable_text}, best validation-to-test transfer={generalizer_text}."
            )
            rows.append(
                {
                    "question_id": "Q10",
                    "question": "Which model offers the best trade-off between peak performance, robustness, and generalization?",
                    "scope": "overall",
                    "answer": synthesis,
                    "supporting_tables": "cross_model_comparison.csv;variability_summary.csv;hpo_model_comparison.csv;hpo_stage_summary.csv",
                    "supporting_plots": "hpo_model_comparison_overview.png;validation_best_vs_final_test.png;best_hota_by_stage.png",
                }
            )

    if not hpo_convergence_summary.empty:
        best_improvement_df = (
            improvement_df.loc[improvement_df["comparison_type"] == "best"].copy()
            if not improvement_df.empty
            else pd.DataFrame()
        )
        for _, row in hpo_convergence_summary.sort_values("model").iterrows():
            model = row["model"]
            model_improvements = (
                best_improvement_df.loc[best_improvement_df["model"] == model]
                if not best_improvement_df.empty
                else pd.DataFrame()
            )
            ft_to_hpo = model_improvements.loc[
                (model_improvements["from_stage"] == "finetuning")
                & (model_improvements["to_stage"] == "hyperparameter_tuning")
            ]
            base_to_hpo = model_improvements.loc[
                (model_improvements["from_stage"] == "baseline")
                & (model_improvements["to_stage"] == "hyperparameter_tuning")
            ]
            if not ft_to_hpo.empty:
                stage_text = (
                    f"finetuning-to-HPO change was {format_optional(ft_to_hpo.iloc[0]['absolute_improvement'])} HOTA"
                )
            elif not base_to_hpo.empty:
                stage_text = (
                    f"baseline-to-HPO change was {format_optional(base_to_hpo.iloc[0]['absolute_improvement'])} HOTA"
                )
            else:
                stage_text = "no direct HPO gain estimate was available"
            rows.append(
                {
                    "question_id": "Q11",
                    "question": "Does the model saturate, or is there still room for improvement?",
                    "scope": model,
                    "answer": (
                        f"For {model}, {stage_text}; {row['convergence_comment'].lower()} "
                        f"The search reached 95% of best by trial {int(row['first_trial_reaching_95pct_of_best'])}."
                    ),
                    "supporting_tables": "improvement_summary.csv;hpo_convergence_summary.csv;hpo_stage_summary.csv",
                    "supporting_plots": "hpo_trial_progression_by_model.png;hpo_convergence_summary_overview.png;hota_improvement_across_stages.png",
                }
            )

    if not descriptive_df.empty:
        stage_presence = (
            descriptive_df.loc[:, ["model", "stage"]]
            .drop_duplicates()
            .groupby("model")["stage"]
            .apply(lambda values: sorted(values, key=lambda item: STAGE_ORDER.get(item, 99)))
        )
        common_stages = set.intersection(
            *(set(stages) for stages in stage_presence.tolist())
        ) if len(stage_presence) > 0 else set()
        inference_note = None
        if not statistical_df.empty and "test_status" in statistical_df.columns:
            if statistical_df["test_status"].astype(str).eq("descriptive_only").all():
                inference_note = "Inferential tests remained descriptive-only because most non-HPO stages have very small sample sizes."
        stage_bits = [f"{model}={', '.join(stages)}" for model, stages in stage_presence.items()]
        rows.append(
            {
                "question_id": "Q12",
                "question": "How fair and informative are the cross-model comparisons?",
                "scope": "overall",
                "answer": (
                    f"Cross-model HOTA comparisons share common stages {', '.join(sorted(common_stages, key=lambda item: STAGE_ORDER.get(item, 99)))}. "
                    f"Model stage coverage was {'; '.join(stage_bits)}. "
                    f"Best-run and final-evaluation comparisons are the most informative; mean-based non-HPO comparisons remain limited where n=1."
                    + (f" {inference_note}" if inference_note else "")
                ),
                "supporting_tables": "cross_model_comparison.csv;descriptive_summary.csv;raw_runs_export.csv;cleaned_runs.csv;statistical_tests.csv",
                "supporting_plots": "best_hota_by_stage.png",
            }
        )

    return pd.DataFrame(rows)


def build_hyperparameter_question_answers(
    sensitivity_df: pd.DataFrame,
    hpo_parameter_group_tests: pd.DataFrame,
    hpo_stage_summary: pd.DataFrame,
    hpo_top_trials: pd.DataFrame,
    hpo_model_comparison: pd.DataFrame,
    cross_model_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not sensitivity_df.empty:
        for model in sorted(sensitivity_df["model"].dropna().unique()):
            model_sens = sensitivity_df.loc[sensitivity_df["model"] == model].copy()
            if model_sens.empty:
                continue
            numeric_ranked = model_sens.loc[
                model_sens["parameter_type"] == "numeric"
            ].sort_values(["association_strength", "parameter_type_rank"], ascending=[False, True])
            categorical_ranked = model_sens.loc[
                model_sens["parameter_type"] == "categorical"
            ].sort_values(["association_strength", "parameter_type_rank"], ascending=[False, True])
            param_bits: list[str] = []
            for _, row in categorical_ranked.head(2).iterrows():
                param_bits.append(
                    f"{row['display_parameter']} (median-spread={format_optional(row.get('association_strength'))}; {row.get('interpretation', '').strip()})"
                )
            for _, row in numeric_ranked.head(3).iterrows():
                param_bits.append(
                    f"{row['display_parameter']} (|Spearman rho|={format_optional(row.get('association_strength'))}; {row.get('interpretation', '').strip()})"
                )
            rows.append(
                {
                    "question_id": "Q1",
                    "question": "Which hyperparameters most strongly influence HOTA?",
                    "scope": model,
                    "answer": (
                        f"For {model}, the strongest categorical HOTA drivers and strongest numeric HOTA drivers were "
                        + "; ".join(param_bits)
                        + "."
                    ),
                    "supporting_tables": "hyperparameter_sensitivity.csv;hpo_parameter_group_tests.csv",
                    "supporting_plots": "hpo_parameter_group_tests_overview.png",
                }
            )

            numeric = numeric_ranked.head(3)
            for _, row in numeric.iterrows():
                direction = (
                    "positive"
                    if pd.notna(row.get("spearman_rho")) and float(row.get("spearman_rho")) > 0
                    else "negative"
                )
                rows.append(
                    {
                        "question_id": "Q2",
                        "question": "In which direction do the important numeric hyperparameters affect HOTA?",
                        "scope": f"{model}: {row['display_parameter']}",
                        "answer": (
                            f"For {model}, {row['display_parameter']} showed a {direction} monotonic association "
                            f"with HOTA (Spearman rho={format_optional(row.get('spearman_rho'))})."
                        ),
                        "supporting_tables": "hyperparameter_sensitivity.csv",
                        "supporting_plots": f"{slugify(model)}_{slugify(row['display_parameter'])}_scatter.png;{slugify(model)}_hyperparameter_correlation_heatmap.png",
                    }
                )
                if pd.notna(row.get("suggested_best_range")):
                    rows.append(
                        {
                            "question_id": "Q3",
                            "question": "Which value ranges appear most promising?",
                            "scope": f"{model}: {row['display_parameter']}",
                            "answer": (
                                f"For {model}, the most promising observed range for {row['display_parameter']} was "
                                f"{row['suggested_best_range']}."
                            ),
                            "supporting_tables": "hyperparameter_sensitivity.csv;hpo_top_trials.csv",
                            "supporting_plots": f"{slugify(model)}_{slugify(row['display_parameter'])}_scatter.png",
                        }
                    )

    if not hpo_parameter_group_tests.empty:
        categorical = hpo_parameter_group_tests.loc[
            hpo_parameter_group_tests["parameter_type"] == "categorical"
        ].copy()
        if not categorical.empty:
            for model in sorted(categorical["model"].dropna().unique()):
                model_cat = categorical.loc[categorical["model"] == model].sort_values(
                    ["ranking_score", "p_value"], ascending=[False, True], na_position="last"
                )
                best_row = model_cat.iloc[0]
                rows.append(
                    {
                        "question_id": "Q4",
                        "question": "Which categorical settings are associated with the best HOTA?",
                        "scope": model,
                        "answer": (
                            f"For {model}, the strongest categorical setting signal came from {best_row['display_parameter']}, "
                            f"where best group {best_row['best_group_label']} reached median HOTA "
                            f"{format_optional(best_row.get('best_group_median_HOTA'))}."
                        ),
                        "supporting_tables": "hpo_parameter_group_tests.csv;hyperparameter_sensitivity.csv",
                        "supporting_plots": "hpo_parameter_group_tests_overview.png",
                    }
                )

    if not cross_model_df.empty:
        ranked = cross_model_df.sort_values("model")
        if not ranked.empty:
            for _, row in ranked.iterrows():
                rows.append(
                    {
                        "question_id": "Q5",
                        "question": "How sensitive is each model to hyperparameter choice overall?",
                        "scope": row["model"],
                        "answer": (
                            f"{row['model']} showed numeric hyperparameter sensitivity up to "
                            f"{format_optional(row.get('numeric_hyperparameter_sensitivity_score'))} "
                            f"(maximum |Spearman rho|) and categorical sensitivity up to "
                            f"{format_optional(row.get('categorical_hyperparameter_sensitivity_score'))} "
                            f"(largest median HOTA spread between groups). "
                            f"These are descriptive, single-parameter sensitivity summaries rather than a multivariate global sensitivity index."
                        ),
                        "supporting_tables": "cross_model_comparison.csv;hpo_model_comparison.csv;hpo_stage_summary.csv",
                        "supporting_plots": "hpo_model_comparison_overview.png;hpo_stage_summary_overview.png;hpo_hota_ecdf_by_model.png",
                    }
                )

    if not hpo_stage_summary.empty:
        for _, row in hpo_stage_summary.sort_values("model").iterrows():
            broad_or_fragile = (
                "broad"
                if pd.notna(row.get("proportion_within_10pct_of_best"))
                and float(row.get("proportion_within_10pct_of_best")) >= 0.30
                else "fragile"
            )
            rows.append(
                {
                    "question_id": "Q6",
                    "question": "Is high performance broad or fragile?",
                    "scope": row["model"],
                    "answer": (
                        f"For {row['model']}, high HOTA looked {broad_or_fragile}: best-minus-median was "
                        f"{format_optional(row.get('best_minus_median_HOTA'))} and "
                        f"{format_optional((row.get('proportion_within_10pct_of_best') or 0) * 100 if pd.notna(row.get('proportion_within_10pct_of_best')) else None)}% "
                        f"of trials were within 10% of the best."
                    ),
                    "supporting_tables": "hpo_stage_summary.csv;hpo_top_trials.csv",
                    "supporting_plots": "hpo_stage_summary_overview.png;hpo_hota_ecdf_by_model.png",
                }
            )

    if not hpo_parameter_group_tests.empty:
        sorted_tests = hpo_parameter_group_tests.sort_values(
            ["model", "ranking_score", "p_value"],
            ascending=[True, False, True],
            na_position="last",
        )
        for model in sorted(sorted_tests["model"].dropna().unique()):
            model_tests = sorted_tests.loc[sorted_tests["model"] == model]
            if model_tests.empty:
                continue
            unstable = model_tests.iloc[0]
            spread = unstable.get("group_spread_HOTA")
            if pd.isna(spread):
                continue
            rows.append(
                {
                    "question_id": "Q7",
                    "question": "Which hyperparameters seem to drive instability?",
                    "scope": model,
                    "answer": (
                        f"For {model}, {unstable['display_parameter']} showed the largest observed grouped HOTA spread "
                        f"({format_optional(spread)}), suggesting a comparatively unstable region of the search space."
                    ),
                    "supporting_tables": "hpo_parameter_group_tests.csv;hpo_stage_summary.csv;hyperparameter_sensitivity.csv",
                    "supporting_plots": "hpo_parameter_group_tests_overview.png;hpo_hota_ecdf_by_model.png",
                }
            )

    if not sensitivity_df.empty:
        top_by_model: list[str] = []
        ranked = sensitivity_df.sort_values(
            ["model", "parameter_type", "association_strength", "parameter_type_rank"],
            ascending=[True, True, False, True],
        )
        for model in sorted(ranked["model"].dropna().unique()):
            model_rows = ranked.loc[ranked["model"] == model]
            if model_rows.empty:
                continue
            numeric_top = model_rows.loc[model_rows["parameter_type"] == "numeric"].head(1)
            categorical_top = model_rows.loc[model_rows["parameter_type"] == "categorical"].head(1)
            parts: list[str] = []
            if not categorical_top.empty:
                parts.append(f"categorical {categorical_top.iloc[0]['display_parameter']}")
            if not numeric_top.empty:
                parts.append(f"numeric {numeric_top.iloc[0]['display_parameter']}")
            if parts:
                top_by_model.append(f"{model}=" + ", ".join(parts))
        if top_by_model:
            rows.append(
                {
                    "question_id": "Q8",
                    "question": "Are the hyperparameter effects model-specific or shared across models?",
                    "scope": "cross_model",
                    "answer": (
                        "The strongest hyperparameter signals appeared model-specific rather than shared: "
                        + "; ".join(top_by_model)
                        + "."
                    ),
                    "supporting_tables": "hyperparameter_sensitivity.csv;hpo_parameter_group_tests.csv;cross_model_comparison.csv",
                    "supporting_plots": "hpo_parameter_group_tests_overview.png",
                }
            )

    return pd.DataFrame(rows)


def build_variability_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, stage), group in cleaned_df.groupby(
        ["model", "stage"], dropna=False, sort=False
    ):
        hota = maybe_numeric(group["HOTA"]).dropna()
        if hota.empty:
            continue
        q1 = float(hota.quantile(0.25))
        q3 = float(hota.quantile(0.75))
        mean = float(hota.mean())
        std = float(hota.std(ddof=1)) if len(hota) > 1 else None
        coefficient_of_variation = (
            (std / mean) if std is not None and mean not in (0, None) else None
        )
        rows.append(
            {
                "model": model,
                "stage": stage,
                "n_runs": int(hota.count()),
                "std_HOTA": std,
                "variance_HOTA": float(hota.var(ddof=1)) if len(hota) > 1 else None,
                "coefficient_of_variation": coefficient_of_variation,
                "range_HOTA": float(hota.max() - hota.min()),
                "iqr_HOTA": q3 - q1,
                "robustness_comment": classify_variability(
                    std, coefficient_of_variation
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["stage_order"] = summary["stage"].map(STAGE_ORDER)
    return (
        summary.sort_values(["model", "stage_order"])
        .drop(columns=["stage_order"])
        .reset_index(drop=True)
    )


def classify_variability(
    std: float | None, coefficient_of_variation: float | None
) -> str:
    if std is None or coefficient_of_variation is None:
        return "Insufficient repeated runs for variability assessment."
    if coefficient_of_variation < 0.02:
        return "Very stable HOTA outcomes across runs."
    if coefficient_of_variation < 0.05:
        return "Moderately stable HOTA outcomes across runs."
    return "High relative variability suggests sensitivity or instability."


def get_candidate_param_columns(
    hpo_df: pd.DataFrame, model: str, config: dict[str, Any]
) -> list[str]:
    preferred_columns = [
        f"hp_{name}" for name in config["preferred_hyperparameters"].get(model, [])
    ]
    additional_columns: list[str] = []
    for column in hpo_df.columns:
        if column.startswith("param.") and not column.startswith("param.mlflow."):
            series = hpo_df[column].dropna()
            if series.nunique() >= 2:
                additional_columns.append(column)
    ordered = preferred_columns + sorted(
        column for column in additional_columns if column not in preferred_columns
    )
    return [column for column in ordered if column in hpo_df.columns]


def prettify_parameter_name(parameter_name: str) -> str:
    cleaned = parameter_name.removeprefix("hp_").removeprefix("param.")
    if cleaned.startswith("optuna."):
        cleaned = cleaned.split(".", 1)[1]
    return cleaned.replace("_", " ").replace(".", " ").strip()


def is_analysis_parameter_name(parameter_name: str) -> bool:
    upper_name = parameter_name.upper()
    excluded_tokens = {
        "EXP_NAME",
        "OUTPUTS_DIR",
        "RUN_NAME",
        "PARENT_RUN_ID",
        "HPO_PARENT_RUN_ID",
        "MLFLOW_RUN_NAME",
        "HPO_STAGE_ITER",
        "HPO_TRIAL_NUMBER",
        "SEED",
    }
    return upper_name not in excluded_tokens


def canonical_parameter_label(
    column: str, model: str, config: dict[str, Any]
) -> str:
    raw_name = column.removeprefix("hp_").removeprefix("param.")
    mapped = config["param_name_mapping"].get(model, {}).get(raw_name)
    if mapped:
        return prettify_parameter_name(mapped).lower()
    return prettify_parameter_name(raw_name).lower()


def deduplicate_parameter_columns(
    columns: list[str], model: str, config: dict[str, Any]
) -> list[str]:
    ranked_columns = sorted(
        columns,
        key=lambda column: (
            0 if column.startswith("hp_") else 1,
            0 if not column.startswith("param.optuna.") else 1,
            column,
        ),
    )
    seen_labels: set[str] = set()
    deduped: list[str] = []
    for column in ranked_columns:
        display_label = canonical_parameter_label(column, model, config)
        if display_label in seen_labels:
            continue
        seen_labels.add(display_label)
        deduped.append(column)
    return deduped


def infer_numeric_range(series: pd.Series, hota: pd.Series) -> str | None:
    data = pd.DataFrame(
        {"param": maybe_numeric(series), "HOTA": maybe_numeric(hota)}
    ).dropna()
    if len(data) < 4 or data["param"].nunique() < 3:
        return None
    threshold = data["HOTA"].quantile(0.75)
    top_slice = data.loc[data["HOTA"] >= threshold, "param"]
    if top_slice.empty:
        return None
    return f"{top_slice.min():.6g} to {top_slice.max():.6g}"


def interpret_numeric_association(
    spearman_rho: float | None, p_value: float | None
) -> str:
    if spearman_rho is None:
        return "Numeric relationship was not estimable."
    magnitude = abs(spearman_rho)
    if magnitude >= 0.7:
        strength = "strong"
    elif magnitude >= 0.4:
        strength = "moderate"
    elif magnitude >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "positive" if spearman_rho > 0 else "negative"
    if p_value is not None and p_value > 0.05:
        return f"{strength.capitalize()} {direction} monotonic association with limited statistical support."
    return f"{strength.capitalize()} {direction} monotonic association."


def interpret_categorical_association(spread: float, unique_values: int) -> str:
    if unique_values <= 1:
        return "Insufficient categorical variation."
    if spread >= 5:
        return "Large between-group HOTA differences suggest threshold-like behavior."
    if spread >= 2:
        return "Moderate between-group HOTA differences suggest a meaningful categorical effect."
    if spread > 0:
        return "Only small between-group HOTA differences were observed."
    return "No visible categorical separation was observed."


def build_hyperparameter_sensitivity(
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cleaned_df.empty:
        return pd.DataFrame()

    for model in sorted(cleaned_df["model"].dropna().unique()):
        hpo_ids = set(
            cleaned_df.loc[
                (cleaned_df["model"] == model)
                & (cleaned_df["stage"] == "hyperparameter_tuning"),
                "run_id",
            ]
        )
        if not hpo_ids:
            continue
        hpo_df = raw_df.loc[raw_df["run_id"].isin(hpo_ids)].copy()
        candidate_columns = deduplicate_parameter_columns(
            get_candidate_param_columns(hpo_df, model, config), model, config
        )
        for column in candidate_columns:
            parameter_name = column.removeprefix("hp_").removeprefix("param.")
            if not is_analysis_parameter_name(parameter_name):
                continue
            display_name = prettify_parameter_name(parameter_name)
            values = hpo_df[column]
            numeric_values = maybe_numeric(values)
            numeric_mask = numeric_values.notna()
            hota = maybe_numeric(hpo_df["normalized_HOTA"])

            if numeric_mask.sum() >= 3 and numeric_values[numeric_mask].nunique() >= 3:
                corr_df = pd.DataFrame({"param": numeric_values, "HOTA": hota}).dropna()
                if len(corr_df) >= 3:
                    pearson_r, pearson_p = stats.pearsonr(
                        corr_df["param"], corr_df["HOTA"]
                    )
                    spearman_rho, spearman_p = stats.spearmanr(
                        corr_df["param"], corr_df["HOTA"]
                    )
                    rows.append(
                        {
                            "model": model,
                            "parameter": parameter_name,
                            "display_parameter": display_name,
                            "source_column": column,
                            "parameter_type": "numeric",
                            "n_observations": int(len(corr_df)),
                            "n_unique_values": int(corr_df["param"].nunique()),
                            "pearson_r": float(pearson_r),
                            "pearson_p_value": float(pearson_p),
                            "spearman_rho": float(spearman_rho),
                            "spearman_p_value": float(spearman_p),
                            "best_category": None,
                            "best_category_mean_HOTA": None,
                            "best_category_median_HOTA": None,
                            "suggested_best_range": infer_numeric_range(
                                corr_df["param"], corr_df["HOTA"]
                            ),
                            "association_strength": abs(float(spearman_rho)),
                            "association_basis": "absolute_spearman_rho",
                            "interpretation": interpret_numeric_association(
                                float(spearman_rho), float(spearman_p)
                            ),
                            "notes": "Numeric importance is ranked by absolute Spearman correlation against normalized HOTA.",
                        }
                    )
                    continue

            grouped = (
                pd.DataFrame({"param": values.astype(str), "HOTA": hota})
                .dropna()
                .groupby("param")
                .agg(
                    count=("HOTA", "count"),
                    mean_HOTA=("HOTA", "mean"),
                    median_HOTA=("HOTA", "median"),
                    max_HOTA=("HOTA", "max"),
                )
                .reset_index()
            )
            if len(grouped) < 2:
                continue
            median_spread = float(
                grouped["median_HOTA"].max() - grouped["median_HOTA"].min()
            )
            best_group = grouped.sort_values(
                ["median_HOTA", "count"], ascending=[False, False]
            ).iloc[0]
            rows.append(
                {
                    "model": model,
                    "parameter": parameter_name,
                    "display_parameter": display_name,
                    "source_column": column,
                    "parameter_type": "categorical",
                    "n_observations": int(grouped["count"].sum()),
                    "n_unique_values": int(len(grouped)),
                    "pearson_r": None,
                    "pearson_p_value": None,
                    "spearman_rho": None,
                    "spearman_p_value": None,
                    "best_category": best_group["param"],
                    "best_category_mean_HOTA": float(best_group["mean_HOTA"]),
                    "best_category_median_HOTA": float(best_group["median_HOTA"]),
                    "suggested_best_range": None,
                    "association_strength": median_spread,
                    "association_basis": "median_hota_spread",
                    "interpretation": interpret_categorical_association(
                        median_spread,
                        int(len(grouped)),
                    ),
                    "notes": "Categorical importance is ranked by spread between the largest and smallest group medians against normalized HOTA.",
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["model", "parameter_type", "association_strength"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    result["parameter_type_rank"] = result.groupby(
        ["model", "parameter_type"]
    ).cumcount() + 1
    result["model_rank"] = result["parameter_type_rank"]
    return result


def get_hpo_analysis_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    hpo_df = cleaned_df.loc[cleaned_df["stage"] == "hyperparameter_tuning"].copy()
    if hpo_df.empty:
        return hpo_df
    hpo_df["HOTA"] = maybe_numeric(hpo_df["HOTA"])
    hpo_df["hpo_trial_number_numeric"] = maybe_numeric(hpo_df["hpo_trial_number"])
    hpo_df["start_time_dt"] = pd.to_datetime(hpo_df["start_time"], errors="coerce")
    hpo_df["trial_order"] = (
        hpo_df.sort_values(["model", "hpo_trial_number_numeric", "start_time_dt", "run_id"])
        .groupby("model")
        .cumcount()
        + 1
    )
    return hpo_df


def build_hpo_stage_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    rows: list[dict[str, Any]] = []
    if hpo_df.empty:
        return pd.DataFrame()

    baseline_lookup = (
        cleaned_df.loc[cleaned_df["stage"] == "baseline", ["model", "HOTA"]]
        .dropna(subset=["HOTA"])
        .groupby("model")["HOTA"]
        .max()
        .to_dict()
    )
    finetuning_lookup = (
        cleaned_df.loc[cleaned_df["stage"] == "finetuning", ["model", "HOTA"]]
        .dropna(subset=["HOTA"])
        .groupby("model")["HOTA"]
        .max()
        .to_dict()
    )

    for model, group in hpo_df.groupby("model", sort=False):
        hota = maybe_numeric(group["HOTA"]).dropna()
        if hota.empty:
            continue
        best = float(hota.max())
        baseline = baseline_lookup.get(model)
        finetuning = finetuning_lookup.get(model)
        top_decile_threshold = float(hota.quantile(0.9))
        top_quartile_threshold = float(hota.quantile(0.75))
        rows.append(
            {
                "model": model,
                "n_hpo_trials": int(hota.count()),
                "mean_HOTA": float(hota.mean()),
                "median_HOTA": float(hota.median()),
                "std_HOTA": float(hota.std(ddof=1)) if len(hota) > 1 else None,
                "variance_HOTA": float(hota.var(ddof=1)) if len(hota) > 1 else None,
                "mad_HOTA": float((hota - hota.median()).abs().median()),
                "iqr_HOTA": float(hota.quantile(0.75) - hota.quantile(0.25)),
                "p10_HOTA": float(hota.quantile(0.10)),
                "p25_HOTA": float(hota.quantile(0.25)),
                "p75_HOTA": float(hota.quantile(0.75)),
                "p90_HOTA": float(hota.quantile(0.90)),
                "best_HOTA": best,
                "top_quartile_mean_HOTA": float(hota[hota >= top_quartile_threshold].mean()),
                "top_decile_mean_HOTA": float(hota[hota >= top_decile_threshold].mean()),
                "best_minus_median_HOTA": best - float(hota.median()),
                "proportion_above_baseline": float((hota > baseline).mean()) if baseline is not None else None,
                "proportion_above_finetuning": float((hota > finetuning).mean()) if finetuning is not None else None,
                "proportion_within_5pct_of_best": float((hota >= best * 0.95).mean()) if best != 0 else None,
                "proportion_within_10pct_of_best": float((hota >= best * 0.90).mean()) if best != 0 else None,
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_hpo_convergence_summary(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    rows: list[dict[str, Any]] = []
    if hpo_df.empty:
        return pd.DataFrame()

    for model, group in hpo_df.groupby("model", sort=False):
        ordered = group.sort_values(["trial_order", "run_id"]).reset_index(drop=True)
        ordered["best_so_far_HOTA"] = ordered["HOTA"].cummax()
        final_best = float(ordered["best_so_far_HOTA"].iloc[-1])
        if final_best > 0:
            first_90 = int(ordered.loc[ordered["best_so_far_HOTA"] >= final_best * 0.90, "trial_order"].iloc[0])
            first_95 = int(ordered.loc[ordered["best_so_far_HOTA"] >= final_best * 0.95, "trial_order"].iloc[0])
        else:
            first_90 = first_95 = int(ordered["trial_order"].iloc[0])
        quartile_index = max(1, int(math.ceil(len(ordered) * 0.25)))
        half_index = max(1, int(math.ceil(len(ordered) * 0.50)))
        rows.append(
            {
                "model": model,
                "n_hpo_trials": int(len(ordered)),
                "best_HOTA": final_best,
                "trial_order_of_best": int(ordered.loc[ordered["HOTA"].idxmax(), "trial_order"]),
                "first_trial_reaching_90pct_of_best": first_90,
                "first_trial_reaching_95pct_of_best": first_95,
                "best_so_far_after_first_quartile": float(
                    ordered.loc[ordered["trial_order"] <= quartile_index, "best_so_far_HOTA"].max()
                ),
                "best_so_far_after_half_trials": float(
                    ordered.loc[ordered["trial_order"] <= half_index, "best_so_far_HOTA"].max()
                ),
                "improvement_last_minus_first_trial": float(ordered["HOTA"].iloc[-1] - ordered["HOTA"].iloc[0]),
                "best_so_far_gain_over_first_trial": float(final_best - ordered["HOTA"].iloc[0]),
                "convergence_comment": (
                    "Best-so-far performance was reached early in the search."
                    if first_95 <= max(3, int(math.ceil(len(ordered) * 0.25)))
                    else "Best-so-far performance required a substantial portion of the search budget."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_hpo_top_trials(cleaned_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    if hpo_df.empty:
        return pd.DataFrame()
    top_trials = (
        hpo_df.sort_values(["model", "HOTA", "trial_order"], ascending=[True, False, True])
        .groupby("model", sort=False)
        .head(top_n)
        .copy()
    )
    top_trials["rank_within_model"] = top_trials.groupby("model").cumcount() + 1
    columns = [
        "model",
        "rank_within_model",
        "run_id",
        "run_name",
        "HOTA",
        "hpo_trial_number",
        "trial_order",
        "normalized_HOTA_source",
    ]
    available = [column for column in columns if column in top_trials.columns]
    return top_trials[available].reset_index(drop=True)


def build_hpo_model_comparison(
    hpo_stage_summary: pd.DataFrame,
    hpo_convergence_summary: pd.DataFrame,
) -> pd.DataFrame:
    if hpo_stage_summary.empty:
        return pd.DataFrame()
    comparison = hpo_stage_summary.merge(hpo_convergence_summary, on="model", how="left", suffixes=("", "_convergence"))
    comparison["is_best_hpo_model"] = comparison["best_HOTA"].eq(comparison["best_HOTA"].max())
    comparison["is_most_robust_hpo_model"] = comparison["iqr_HOTA"].eq(comparison["iqr_HOTA"].min())
    if comparison["first_trial_reaching_95pct_of_best"].notna().any():
        comparison["is_most_hpo_efficient_model"] = comparison["first_trial_reaching_95pct_of_best"].eq(
            comparison["first_trial_reaching_95pct_of_best"].min()
        )
    else:
        comparison["is_most_hpo_efficient_model"] = False
    return comparison.sort_values("model").reset_index(drop=True)


def build_hpo_statistical_analysis(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    rows: list[dict[str, Any]] = []
    if hpo_df.empty:
        return pd.DataFrame()

    baseline_lookup = (
        cleaned_df.loc[cleaned_df["stage"] == "baseline", ["model", "HOTA"]]
        .dropna(subset=["HOTA"])
        .groupby("model")["HOTA"]
        .max()
        .to_dict()
    )
    finetuning_lookup = (
        cleaned_df.loc[cleaned_df["stage"] == "finetuning", ["model", "HOTA"]]
        .dropna(subset=["HOTA"])
        .groupby("model")["HOTA"]
        .max()
        .to_dict()
    )

    for model, group in hpo_df.groupby("model", sort=False):
        ordered = group.sort_values(["trial_order", "run_id"]).reset_index(drop=True)
        hota = maybe_numeric(ordered["HOTA"]).dropna().to_numpy()
        if len(hota) == 0:
            continue

        baseline = baseline_lookup.get(model)
        if baseline is not None:
            prop_low, prop_high = bootstrap_proportion_above_threshold(hota, float(baseline))
            mean_ci_low, mean_ci_high = bootstrap_diff_vs_scalar(hota, float(baseline), stat="mean")
            median_ci_low, median_ci_high = bootstrap_diff_vs_scalar(hota, float(baseline), stat="median")
            rows.append(
                {
                    "model": model,
                    "analysis_name": "hpo_vs_baseline",
                    "comparison_group": "all_hpo_trials",
                    "n_trials": int(len(hota)),
                    "reference_value": float(baseline),
                    "reference_label": "baseline_HOTA",
                    "mean_HOTA": float(hota.mean()),
                    "median_HOTA": float(np.median(hota)),
                    "mean_difference_vs_reference": float(hota.mean() - baseline),
                    "median_difference_vs_reference": float(np.median(hota) - baseline),
                    "proportion_above_reference": float((hota > baseline).mean()),
                    "proportion_above_reference_ci95_low": prop_low,
                    "proportion_above_reference_ci95_high": prop_high,
                    "mean_difference_ci95_low": mean_ci_low,
                    "mean_difference_ci95_high": mean_ci_high,
                    "median_difference_ci95_low": median_ci_low,
                    "median_difference_ci95_high": median_ci_high,
                    "test_statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "effect_size_label": None,
                    "warning_notes": "Reference is a single baseline run, so inference quantifies the HPO distribution relative to a fixed value rather than a replicated stage.",
                }
            )

        finetuning = finetuning_lookup.get(model)
        if finetuning is not None:
            prop_low, prop_high = bootstrap_proportion_above_threshold(hota, float(finetuning))
            mean_ci_low, mean_ci_high = bootstrap_diff_vs_scalar(hota, float(finetuning), stat="mean")
            median_ci_low, median_ci_high = bootstrap_diff_vs_scalar(hota, float(finetuning), stat="median")
            rows.append(
                {
                    "model": model,
                    "analysis_name": "hpo_vs_finetuning",
                    "comparison_group": "all_hpo_trials",
                    "n_trials": int(len(hota)),
                    "reference_value": float(finetuning),
                    "reference_label": "finetuning_HOTA",
                    "mean_HOTA": float(hota.mean()),
                    "median_HOTA": float(np.median(hota)),
                    "mean_difference_vs_reference": float(hota.mean() - finetuning),
                    "median_difference_vs_reference": float(np.median(hota) - finetuning),
                    "proportion_above_reference": float((hota > finetuning).mean()),
                    "proportion_above_reference_ci95_low": prop_low,
                    "proportion_above_reference_ci95_high": prop_high,
                    "mean_difference_ci95_low": mean_ci_low,
                    "mean_difference_ci95_high": mean_ci_high,
                    "median_difference_ci95_low": median_ci_low,
                    "median_difference_ci95_high": median_ci_high,
                    "test_statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "effect_size_label": None,
                    "warning_notes": "Reference is a single finetuning run, so inference quantifies the HPO distribution relative to a fixed value rather than a replicated stage.",
                }
            )

        quantile_groups = [
            ("top_quartile_vs_rest", 0.75),
            ("top_decile_vs_rest", 0.90),
        ]
        for analysis_name, quantile in quantile_groups:
            threshold = float(np.quantile(hota, quantile))
            top_values = hota[hota >= threshold]
            rest_values = hota[hota < threshold]
            if len(top_values) == 0 or len(rest_values) == 0:
                continue
            u_stat, p_value, rank_biserial = mannwhitney_rank_biserial(top_values, rest_values)
            perm_p = permutation_test_mean_difference(top_values, rest_values)
            mean_ci_low, mean_ci_high = bootstrap_mean_diff(rest_values, top_values)
            median_ci_low, median_ci_high = bootstrap_median_diff(top_values, rest_values)
            rows.append(
                {
                    "model": model,
                    "analysis_name": analysis_name,
                    "comparison_group": f"threshold_at_{quantile:.2f}",
                    "n_trials": int(len(hota)),
                    "reference_value": float(rest_values.mean()),
                    "reference_label": "rest_mean_HOTA",
                    "mean_HOTA": float(top_values.mean()),
                    "median_HOTA": float(np.median(top_values)),
                    "mean_difference_vs_reference": float(top_values.mean() - rest_values.mean()),
                    "median_difference_vs_reference": float(np.median(top_values) - np.median(rest_values)),
                    "proportion_above_reference": float((top_values > rest_values.mean()).mean()),
                    "proportion_above_reference_ci95_low": None,
                    "proportion_above_reference_ci95_high": None,
                    "mean_difference_ci95_low": mean_ci_low,
                    "mean_difference_ci95_high": mean_ci_high,
                    "median_difference_ci95_low": median_ci_low,
                    "median_difference_ci95_high": median_ci_high,
                    "test_statistic": u_stat,
                    "p_value": p_value,
                    "effect_size": rank_biserial,
                    "effect_size_label": "rank_biserial_correlation",
                    "warning_notes": (
                        f"Permutation p-value for mean difference: {perm_p:.4f}. "
                        "This comparison is descriptive of search-space concentration rather than an independent treatment contrast."
                    ),
                }
            )

        split_index = max(1, len(ordered) // 2)
        early_values = maybe_numeric(ordered.iloc[:split_index]["HOTA"]).dropna().to_numpy()
        late_values = maybe_numeric(ordered.iloc[split_index:]["HOTA"]).dropna().to_numpy()
        if len(early_values) > 0 and len(late_values) > 0:
            u_stat, p_value, rank_biserial = mannwhitney_rank_biserial(late_values, early_values)
            perm_p = permutation_test_mean_difference(late_values, early_values)
            mean_ci_low, mean_ci_high = bootstrap_mean_diff(early_values, late_values)
            median_ci_low, median_ci_high = bootstrap_median_diff(late_values, early_values)
            rows.append(
                {
                    "model": model,
                    "analysis_name": "late_vs_early_trials",
                    "comparison_group": "second_half_vs_first_half",
                    "n_trials": int(len(hota)),
                    "reference_value": float(early_values.mean()),
                    "reference_label": "early_half_mean_HOTA",
                    "mean_HOTA": float(late_values.mean()),
                    "median_HOTA": float(np.median(late_values)),
                    "mean_difference_vs_reference": float(late_values.mean() - early_values.mean()),
                    "median_difference_vs_reference": float(np.median(late_values) - np.median(early_values)),
                    "proportion_above_reference": float((late_values > early_values.mean()).mean()),
                    "proportion_above_reference_ci95_low": None,
                    "proportion_above_reference_ci95_high": None,
                    "mean_difference_ci95_low": mean_ci_low,
                    "mean_difference_ci95_high": mean_ci_high,
                    "median_difference_ci95_low": median_ci_low,
                    "median_difference_ci95_high": median_ci_high,
                    "test_statistic": u_stat,
                    "p_value": p_value,
                    "effect_size": rank_biserial,
                    "effect_size_label": "rank_biserial_correlation",
                    "warning_notes": (
                        f"Permutation p-value for mean difference: {perm_p:.4f}. "
                        "This checks search-efficiency trends across trial order, not independent replicated conditions."
                    ),
                }
            )

        best_minus_median_ci_low, best_minus_median_ci_high = bootstrap_scalar_difference(
            hota, float(np.median(hota))
        )
        rows.append(
            {
                "model": model,
                "analysis_name": "best_minus_median_gap",
                "comparison_group": "all_hpo_trials",
                "n_trials": int(len(hota)),
                "reference_value": float(np.median(hota)),
                "reference_label": "median_HOTA",
                "mean_HOTA": float(hota.max()),
                "median_HOTA": float(np.median(hota)),
                "mean_difference_vs_reference": float(hota.max() - np.median(hota)),
                "median_difference_vs_reference": 0.0,
                "proportion_above_reference": float((hota > np.median(hota)).mean()),
                "proportion_above_reference_ci95_low": None,
                "proportion_above_reference_ci95_high": None,
                "mean_difference_ci95_low": best_minus_median_ci_low,
                "mean_difference_ci95_high": best_minus_median_ci_high,
                "median_difference_ci95_low": None,
                "median_difference_ci95_high": None,
                "test_statistic": None,
                "p_value": None,
                "effect_size": None,
                "effect_size_label": None,
                "warning_notes": "Bootstrap interval quantifies how sharply the best observed configuration separates from the typical HPO trial.",
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "analysis_name"]).reset_index(drop=True)


def build_hpo_parameter_group_tests(
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    rows: list[dict[str, Any]] = []
    if hpo_df.empty:
        return pd.DataFrame()

    for model in sorted(hpo_df["model"].dropna().unique()):
        hpo_ids = set(hpo_df.loc[hpo_df["model"] == model, "run_id"])
        model_raw = raw_df.loc[raw_df["run_id"].isin(hpo_ids)].copy()
        if model_raw.empty:
            continue
        candidate_columns = deduplicate_parameter_columns(
            get_candidate_param_columns(model_raw, model, config), model, config
        )
        hota = maybe_numeric(model_raw["normalized_HOTA"])
        for column in candidate_columns:
            parameter_name = column.removeprefix("hp_").removeprefix("param.")
            if not is_analysis_parameter_name(parameter_name):
                continue
            display_name = prettify_parameter_name(parameter_name)
            numeric_values = maybe_numeric(model_raw[column])
            numeric_df = pd.DataFrame({"param": numeric_values, "HOTA": hota}).dropna()
            if len(numeric_df) >= 5 and numeric_df["param"].nunique() >= 5:
                spearman_rho, spearman_p = safe_spearman(
                    numeric_df["param"], numeric_df["HOTA"]
                )
                if spearman_rho is not None:
                    permutation_p = permutation_test_mean_difference(
                        numeric_df.loc[numeric_df["param"] >= numeric_df["param"].median(), "HOTA"].to_numpy(),
                        numeric_df.loc[numeric_df["param"] < numeric_df["param"].median(), "HOTA"].to_numpy(),
                    )
                    rows.append(
                        {
                            "model": model,
                            "parameter": parameter_name,
                            "display_parameter": display_name,
                            "source_column": column,
                            "parameter_type": "numeric",
                            "n_observations": int(len(numeric_df)),
                            "n_groups": int(numeric_df["param"].nunique()),
                            "test_name": "spearman_correlation",
                            "test_statistic": spearman_rho,
                            "p_value": spearman_p,
                            "effect_size": spearman_rho,
                            "effect_size_label": "spearman_rho",
                            "ranking_basis": "absolute_spearman_rho",
                            "ranking_score": abs(float(spearman_rho)),
                            "best_group_label": None,
                            "best_group_median_HOTA": None,
                            "group_spread_HOTA": None,
                            "warning_notes": (
                                f"Median split permutation p-value for HOTA difference: {permutation_p:.4f}."
                                if permutation_p is not None
                                else "Permutation test was not estimable."
                            ),
                        }
                    )
                    continue

            grouped = (
                pd.DataFrame({"param": model_raw[column].astype(str), "HOTA": hota})
                .dropna()
                .groupby("param")
                .agg(count=("HOTA", "count"), median_HOTA=("HOTA", "median"))
                .reset_index()
            )
            grouped = grouped.loc[grouped["count"] >= 2].copy()
            if len(grouped) < 2:
                continue
            grouped = grouped.sort_values("param").reset_index(drop=True)
            group_values = [
                pd.DataFrame({"param": model_raw[column].astype(str), "HOTA": hota})
                .dropna()
                .loc[lambda df: df["param"] == label, "HOTA"]
                .to_numpy()
                for label in grouped["param"]
            ]
            test_name = "kruskal_wallis" if len(group_values) > 2 else "mannwhitney_u"
            test_statistic = p_value = effect_size = None
            warning_notes = None
            if len(group_values) == 2:
                test_statistic, p_value, effect_size = mannwhitney_rank_biserial(
                    group_values[0], group_values[1]
                )
                warning_notes = "Effect size is rank-biserial correlation for the two-group comparison."
            else:
                kruskal_stat, kruskal_p = stats.kruskal(*group_values)
                test_statistic = float(kruskal_stat)
                p_value = float(kruskal_p)
                effect_size = float(grouped["median_HOTA"].max() - grouped["median_HOTA"].min())
                warning_notes = "Effect size reports the spread between the largest and smallest group medians."
            best_group = grouped.sort_values(["median_HOTA", "count"], ascending=[False, False]).iloc[0]
            rows.append(
                {
                    "model": model,
                    "parameter": parameter_name,
                    "display_parameter": display_name,
                    "source_column": column,
                    "parameter_type": "categorical",
                    "n_observations": int(grouped["count"].sum()),
                    "n_groups": int(len(grouped)),
                    "test_name": test_name,
                    "test_statistic": test_statistic,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "effect_size_label": (
                        "rank_biserial_correlation"
                        if len(group_values) == 2
                        else "median_spread"
                    ),
                    "ranking_basis": "median_hota_spread",
                    "ranking_score": float(
                        grouped["median_HOTA"].max() - grouped["median_HOTA"].min()
                    ),
                    "best_group_label": best_group["param"],
                    "best_group_median_HOTA": float(best_group["median_HOTA"]),
                    "group_spread_HOTA": float(grouped["median_HOTA"].max() - grouped["median_HOTA"].min()),
                    "warning_notes": warning_notes,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["model", "parameter_type", "ranking_score", "p_value"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_hpo_cross_model_tests(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    rows: list[dict[str, Any]] = []
    if hpo_df.empty:
        return pd.DataFrame()

    model_arrays: dict[str, np.ndarray] = {}
    for model, group in hpo_df.groupby("model", sort=False):
        values = maybe_numeric(group["HOTA"]).dropna().to_numpy()
        if len(values) > 0:
            model_arrays[model] = values

    if len(model_arrays) >= 2:
        overall_stat, overall_p = stats.kruskal(*model_arrays.values())
        rows.append(
            {
                "comparison_type": "overall",
                "model_a": None,
                "model_b": None,
                "n_model_a": int(sum(len(values) for values in model_arrays.values())),
                "n_model_b": None,
                "median_model_a": None,
                "median_model_b": None,
                "mean_difference_b_minus_a": None,
                "median_difference_b_minus_a": None,
                "bootstrap_median_diff_ci95_low": None,
                "bootstrap_median_diff_ci95_high": None,
                "test_name": "kruskal_wallis",
                "test_statistic": float(overall_stat),
                "p_value": float(overall_p),
                "effect_size": None,
                "effect_size_label": None,
                "warning_notes": "Overall test compares the HPO distributions across all available models.",
            }
        )

    models = sorted(model_arrays)
    for index, model_a in enumerate(models):
        for model_b in models[index + 1 :]:
            a = model_arrays[model_a]
            b = model_arrays[model_b]
            u_stat, p_value, rank_biserial = mannwhitney_rank_biserial(b, a)
            ci_low, ci_high = bootstrap_median_diff(b, a)
            rows.append(
                {
                    "comparison_type": "pairwise",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_model_a": int(len(a)),
                    "n_model_b": int(len(b)),
                    "median_model_a": float(np.median(a)),
                    "median_model_b": float(np.median(b)),
                    "mean_difference_b_minus_a": float(b.mean() - a.mean()),
                    "median_difference_b_minus_a": float(np.median(b) - np.median(a)),
                    "bootstrap_median_diff_ci95_low": ci_low,
                    "bootstrap_median_diff_ci95_high": ci_high,
                    "test_name": "mannwhitney_u",
                    "test_statistic": u_stat,
                    "p_value": p_value,
                    "effect_size": rank_biserial,
                    "effect_size_label": "rank_biserial_correlation",
                    "warning_notes": "Pairwise HPO comparison uses the distributions of all tuning trials for each model.",
                }
            )
    return pd.DataFrame(rows)


def pooled_std(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = math.sqrt(
        (((len(a) - 1) * s1**2) + ((len(b) - 1) * s2**2)) / (len(a) + len(b) - 2)
    )
    return pooled if pooled > 0 else None


def hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    pooled = pooled_std(a, b)
    if pooled is None:
        return None
    correction = 1 - (3 / (4 * (len(a) + len(b)) - 9))
    return float(((a.mean() - b.mean()) / pooled) * correction)


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(b) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        samples.append(float(sample_b.mean() - sample_a.mean()))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_single_sample_stat(
    values: np.ndarray,
    stat_fn: Any,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(float(stat_fn(sample)))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_diff_vs_scalar(
    values: np.ndarray,
    reference: float,
    stat: str = "mean",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    if stat == "mean":
        stat_fn = np.mean
    elif stat == "median":
        stat_fn = np.median
    else:
        raise ValueError(f"Unsupported stat '{stat}'.")
    return bootstrap_single_sample_stat(
        values,
        lambda sample: float(stat_fn(sample) - reference),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def bootstrap_proportion_above_threshold(
    values: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    return bootstrap_single_sample_stat(
        values,
        lambda sample: float((sample > threshold).mean()),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def mannwhitney_rank_biserial(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    try:
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None, None, None
    rank_biserial = (2 * float(u_stat) / (len(a) * len(b))) - 1
    return float(u_stat), float(p_value), float(rank_biserial)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    comparisons = 0
    for value_a in a:
        comparisons += int((value_a > b).sum())
        comparisons -= int((value_a < b).sum())
    return float(comparisons / (len(a) * len(b)))


def bootstrap_median_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(a) == 0 or len(b) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        samples.append(float(np.median(sample_a) - np.median(sample_b)))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_scalar_difference(
    values: np.ndarray,
    scalar: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    return bootstrap_single_sample_stat(
        values,
        lambda sample: float(sample.max() - scalar),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def permutation_test_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    combined = np.concatenate([a, b])
    if len(np.unique(combined)) <= 1:
        return 1.0
    observed = abs(float(a.mean() - b.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(combined)
        perm_a = permuted[: len(a)]
        perm_b = permuted[len(a) :]
        if abs(float(perm_a.mean() - perm_b.mean())) >= observed:
            exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


def safe_spearman(values: pd.Series, target: pd.Series) -> tuple[float | None, float | None]:
    data = pd.DataFrame({"x": maybe_numeric(values), "y": maybe_numeric(target)}).dropna()
    if len(data) < 3 or data["x"].nunique() < 3 or data["y"].nunique() < 2:
        return None, None
    rho, p_value = stats.spearmanr(data["x"], data["y"])
    if np.isnan(rho) or np.isnan(p_value):
        return None, None
    return float(rho), float(p_value)


def build_statistical_tests(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_pairs = [
        ("baseline", "finetuning"),
        ("finetuning", "hyperparameter_tuning"),
        ("baseline", "hyperparameter_tuning"),
        ("hyperparameter_tuning", "final_evaluation"),
    ]

    for model in sorted(cleaned_df["model"].dropna().unique()):
        model_df = cleaned_df.loc[cleaned_df["model"] == model]
        for stage_a, stage_b in stage_pairs:
            a = (
                maybe_numeric(model_df.loc[model_df["stage"] == stage_a, "HOTA"])
                .dropna()
                .to_numpy()
            )
            b = (
                maybe_numeric(model_df.loc[model_df["stage"] == stage_b, "HOTA"])
                .dropna()
                .to_numpy()
            )
            if len(a) == 0 or len(b) == 0:
                continue

            note_parts: list[str] = []
            t_stat = t_p = mw_stat = mw_p = effect_size = ci_low = ci_high = None
            comparison_type = "inferential"

            if len(a) >= 2 and len(b) >= 2:
                t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    mw_stat, mw_p = None, None
                effect_size = hedges_g(a, b)
                ci_low, ci_high = bootstrap_mean_diff(a, b)
            else:
                comparison_type = "descriptive_only"
                note_parts.append("Very small sample size; inferential tests omitted.")

            if stage_a == "hyperparameter_tuning" and stage_b == "final_evaluation":
                note_parts.append(
                    "Validation-stage tuned results and final test results serve different purposes."
                )

            rows.append(
                {
                    "model": model,
                    "from_stage": stage_a,
                    "to_stage": stage_b,
                    "comparison_type": comparison_type,
                    "n_from_stage": int(len(a)),
                    "n_to_stage": int(len(b)),
                    "mean_from_stage": float(a.mean()),
                    "mean_to_stage": float(b.mean()),
                    "mean_difference_to_minus_from": float(b.mean() - a.mean()),
                    "welch_t_statistic": (
                        float(t_stat)
                        if t_stat is not None and not np.isnan(t_stat)
                        else None
                    ),
                    "welch_t_p_value": (
                        float(t_p) if t_p is not None and not np.isnan(t_p) else None
                    ),
                    "mannwhitney_u_statistic": (
                        float(mw_stat) if mw_stat is not None else None
                    ),
                    "mannwhitney_u_p_value": float(mw_p) if mw_p is not None else None,
                    "hedges_g": effect_size,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "warning_notes": " ".join(note_parts).strip() or None,
                }
            )
    return pd.DataFrame(rows)


def build_cross_model_comparison(
    descriptive_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = sorted(descriptive_df["model"].dropna().unique())
    for model in models:
        model_desc = descriptive_df.loc[descriptive_df["model"] == model].set_index(
            "stage"
        )
        model_var = variability_df.loc[variability_df["model"] == model]
        model_sens = sensitivity_df.loc[sensitivity_df["model"] == model]
        baseline_best = (
            model_desc.loc["baseline", "max_HOTA"]
            if "baseline" in model_desc.index
            else None
        )
        hpo_best = (
            model_desc.loc["hyperparameter_tuning", "max_HOTA"]
            if "hyperparameter_tuning" in model_desc.index
            else None
        )
        final_best = (
            model_desc.loc["final_evaluation", "max_HOTA"]
            if "final_evaluation" in model_desc.index
            else None
        )
        final_mean = (
            model_desc.loc["final_evaluation", "mean_HOTA"]
            if "final_evaluation" in model_desc.index
            else None
        )
        stability_score = None
        if not model_var.empty and model_var["coefficient_of_variation"].notna().any():
            stability_score = float(
                model_var["coefficient_of_variation"].dropna().mean()
            )
        numeric_sensitivity_score = None
        categorical_sensitivity_score = None
        if not model_sens.empty:
            numeric_rows = model_sens.loc[
                model_sens["parameter_type"] == "numeric", "association_strength"
            ].dropna()
            categorical_rows = model_sens.loc[
                model_sens["parameter_type"] == "categorical", "association_strength"
            ].dropna()
            if not numeric_rows.empty:
                numeric_sensitivity_score = float(numeric_rows.max())
            if not categorical_rows.empty:
                categorical_sensitivity_score = float(categorical_rows.max())

        rows.append(
            {
                "model": model,
                "baseline_best_HOTA": baseline_best,
                "finetuning_best_HOTA": (
                    model_desc.loc["finetuning", "max_HOTA"]
                    if "finetuning" in model_desc.index
                    else None
                ),
                "hyperparameter_tuning_best_HOTA": hpo_best,
                "final_evaluation_best_HOTA": final_best,
                "final_evaluation_mean_HOTA": final_mean,
                "baseline_to_tuned_validation_gain": (
                    (hpo_best - baseline_best)
                    if baseline_best is not None and hpo_best is not None
                    else None
                ),
                "baseline_to_final_test_gain": (
                    (final_best - baseline_best)
                    if baseline_best is not None and final_best is not None
                    else None
                ),
                "stability_score_cv_mean": stability_score,
                "numeric_hyperparameter_sensitivity_score": numeric_sensitivity_score,
                "categorical_hyperparameter_sensitivity_score": categorical_sensitivity_score,
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    def mark_best(column: str, ascending: bool = False) -> pd.Series:
        if comparison[column].dropna().empty:
            return pd.Series([False] * len(comparison))
        target = comparison[column].min() if ascending else comparison[column].max()
        return comparison[column].eq(target)

    comparison["is_best_baseline_model"] = mark_best("baseline_best_HOTA")
    comparison["is_most_improved_model"] = mark_best(
        "baseline_to_tuned_validation_gain"
    )
    comparison["is_best_final_model"] = mark_best("final_evaluation_best_HOTA")
    comparison["is_most_stable_model"] = mark_best(
        "stability_score_cv_mean", ascending=True
    )
    comparison["is_most_numeric_sensitive_model"] = mark_best(
        "numeric_hyperparameter_sensitivity_score"
    )
    comparison["is_most_categorical_sensitive_model"] = mark_best(
        "categorical_hyperparameter_sensitivity_score"
    )
    return comparison.sort_values("model").reset_index(drop=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_grouped_parameter_table(
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    model: str,
    display_parameter: str,
) -> pd.DataFrame:
    row = sensitivity_df.loc[
        (sensitivity_df["model"] == model)
        & (
            sensitivity_df["display_parameter"].astype(str).str.casefold()
            == display_parameter.casefold()
        )
    ]
    if row.empty:
        return pd.DataFrame()
    source_column = row.iloc[0]["source_column"]
    hpo_ids = set(
        cleaned_df.loc[
            (cleaned_df["model"] == model)
            & (cleaned_df["stage"] == "hyperparameter_tuning"),
            "run_id",
        ]
    )
    model_raw = raw_df.loc[raw_df["run_id"].isin(hpo_ids)].copy()
    if model_raw.empty or source_column not in model_raw.columns:
        return pd.DataFrame()
    grouped = (
        pd.DataFrame(
            {
                "parameter_value": model_raw[source_column].astype(str),
                "HOTA": maybe_numeric(model_raw["normalized_HOTA"]),
            }
        )
        .dropna()
        .groupby("parameter_value")
        .agg(mean_HOTA=("HOTA", "mean"), count=("HOTA", "count"))
        .reset_index()
        .sort_values("mean_HOTA", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )
    if grouped.empty:
        return grouped
    grouped.insert(0, "model", model)
    grouped.insert(1, "display_parameter", display_parameter)
    return grouped


def export_plot_tables(
    artifacts: PipelineArtifacts,
    output_dir: Path,
) -> None:
    tables_dir = output_dir / "plot_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    descriptive_df = artifacts.descriptive_summary.copy()
    improvement_df = artifacts.improvement_summary.copy()
    raw_df = artifacts.raw_runs.copy()
    cleaned_df = artifacts.cleaned_runs.copy()
    sensitivity_df = artifacts.hyperparameter_sensitivity.copy()

    best_hota_table = descriptive_df.loc[
        descriptive_df["max_HOTA"].notna(), ["model", "stage", "max_HOTA"]
    ].copy()
    write_dataframe(best_hota_table, tables_dir / "best_hota_by_stage_table.csv")

    validation_vs_final_table = descriptive_df.loc[
        descriptive_df["stage"].isin(["hyperparameter_tuning", "final_evaluation"]),
        ["model", "stage", "max_HOTA"],
    ].dropna(subset=["max_HOTA"])
    write_dataframe(
        validation_vs_final_table,
        tables_dir / "validation_best_vs_final_test_table.csv",
    )

    improvement_table = improvement_df.loc[
        improvement_df["comparison_type"] == "best"
    ].copy()
    if not improvement_table.empty:
        improvement_table["comparison"] = (
            improvement_table["from_stage"] + " -> " + improvement_table["to_stage"]
        )
        improvement_table = improvement_table[
            [
                "model",
                "comparison",
                "from_stage",
                "to_stage",
                "from_HOTA",
                "to_HOTA",
                "absolute_improvement",
                "relative_improvement_pct",
            ]
        ]
    write_dataframe(improvement_table, tables_dir / "improvement_across_stages_table.csv")

    decomposition_improvement_table = artifacts.decomposition_improvement_summary.copy()
    if not decomposition_improvement_table.empty:
        decomposition_improvement_table["comparison"] = (
            decomposition_improvement_table["from_stage"]
            + " -> "
            + decomposition_improvement_table["to_stage"]
        )
        comparison_order = {
            "baseline -> finetuning": 0,
            "baseline -> hyperparameter_tuning": 1,
            "baseline -> final_evaluation": 2,
            "finetuning -> hyperparameter_tuning": 3,
            "hyperparameter_tuning -> final_evaluation": 4,
        }
        decomposition_improvement_table["comparison_order"] = decomposition_improvement_table[
            "comparison"
        ].map(comparison_order)
        decomposition_improvement_table = decomposition_improvement_table.sort_values(
            ["model", "comparison_order"]
        )[
            [
                "model",
                "comparison",
                "delta_DetA",
                "delta_AssA",
                "delta_DetPr",
                "delta_DetRe",
                "delta_AssPr",
                "delta_AssRe",
            ]
        ]
    write_dataframe(
        decomposition_improvement_table,
        tables_dir / "decomposition_improvements_table.csv",
    )

    hpo_stage_perf = artifacts.hpo_stage_summary.loc[
        :,
        [
            "model",
            "median_HOTA",
            "best_HOTA",
            "top_quartile_mean_HOTA",
            "proportion_above_baseline",
            "proportion_within_5pct_of_best",
        ],
    ].copy()
    write_dataframe(hpo_stage_perf, tables_dir / "hpo_stage_summary_overview_table.csv")

    hpo_convergence_table = artifacts.hpo_convergence_summary.loc[
        :,
        [
            "model",
            "first_trial_reaching_90pct_of_best",
            "first_trial_reaching_95pct_of_best",
            "best_so_far_gain_over_first_trial",
            "improvement_last_minus_first_trial",
        ],
    ].copy()
    write_dataframe(
        hpo_convergence_table,
        tables_dir / "hpo_convergence_summary_overview_table.csv",
    )

    hpo_top_trials_table = artifacts.hpo_top_trials.copy()
    write_dataframe(hpo_top_trials_table, tables_dir / "hpo_top_trials_by_model_table.csv")

    hpo_model_comparison_table = artifacts.hpo_model_comparison.loc[
        :,
        [
            "model",
            "best_HOTA",
            "median_HOTA",
            "top_decile_mean_HOTA",
            "iqr_HOTA",
            "first_trial_reaching_95pct_of_best",
            "is_best_hpo_model",
            "is_most_robust_hpo_model",
            "is_most_hpo_efficient_model",
        ],
    ].copy()
    write_dataframe(
        hpo_model_comparison_table,
        tables_dir / "hpo_model_comparison_overview_table.csv",
    )

    hpo_parameter_table = artifacts.hpo_parameter_group_tests.copy()
    if not hpo_parameter_table.empty:
        if "ranking_score" not in hpo_parameter_table.columns:
            hpo_parameter_table["ranking_score"] = np.where(
                hpo_parameter_table["parameter_type"].eq("numeric"),
                hpo_parameter_table["effect_size"].abs(),
                hpo_parameter_table["group_spread_HOTA"],
            )
        hpo_parameter_table = (
            hpo_parameter_table.sort_values(
                ["model", "parameter_type", "ranking_score"],
                ascending=[True, True, False],
            )
            .groupby("model", sort=False)
            .head(5)
            .reset_index(drop=True)
        )
    write_dataframe(
        hpo_parameter_table,
        tables_dir / "hpo_parameter_group_tests_overview_table.csv",
    )

    motip_area_thresh_table = build_grouped_parameter_table(
        cleaned_df, raw_df, sensitivity_df, "MOTIP", "AREA THRESH"
    )
    write_dataframe(
        motip_area_thresh_table,
        tables_dir / "motip_area_thresh_grouped_table.csv",
    )

    motip_assignment_protocol_table = build_grouped_parameter_table(
        cleaned_df, raw_df, sensitivity_df, "MOTIP", "assignment protocol"
    )
    write_dataframe(
        motip_assignment_protocol_table,
        tables_dir / "motip_assignment_protocol_grouped_table.csv",
    )


def format_table_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def render_table_plot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    col_widths: list[float] | None = None,
) -> None:
    if df.empty:
        return
    display_df = df.copy()
    cell_text = [[format_table_value(value) for value in row] for row in display_df.to_numpy()]
    n_rows, n_cols = display_df.shape
    fig_width = max(8, n_cols * 1.8)
    fig_height = max(2.6, (n_rows + 1) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    row_colors = [sns.color_palette("light:#4C72B0", n_colors=3)[1], "white"]
    table = ax.table(
        cellText=cell_text,
        colLabels=[str(column) for column in display_df.columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_best_hota_stage_matrix_plot(
    descriptive_df: pd.DataFrame,
    output_path: Path,
) -> None:
    data = descriptive_df.loc[
        descriptive_df["max_HOTA"].notna(), ["model", "stage", "max_HOTA"]
    ].copy()
    if data.empty:
        return

    stage_labels = {
        "baseline": "Baseline",
        "finetuning": "Finetuning",
        "hyperparameter_tuning": "HPT (Best Trial)",
        "final_evaluation": "Final Evaluation (Test)",
    }
    ordered_stages = [stage for stage in NORMALIZED_STAGES if stage in data["stage"].unique()]
    ordered_models = sorted(data["model"].dropna().unique())
    matrix = (
        data.pivot(index="model", columns="stage", values="max_HOTA")
        .reindex(index=ordered_models)
        .reindex(columns=ordered_stages)
    )
    display_df = matrix.copy()
    display_df.columns = [stage_labels.get(stage, stage) for stage in display_df.columns]
    formatted_df = display_df.applymap(
        lambda value: "—" if pd.isna(value) else f"{float(value):.2f}"
    )

    n_rows, n_cols = formatted_df.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 2.3), max(3.2, (n_rows + 1) * 0.7)))
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    neutral_row = sns.color_palette("light:#4C72B0", n_colors=4)[1]
    best_cell_color = sns.color_palette("deep")[2]
    missing_cell_color = "#F3F3F3"

    table = ax.table(
        cellText=formatted_df.to_numpy().tolist(),
        rowLabels=list(formatted_df.index),
        colLabels=[str(column) for column in formatted_df.columns],
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.5)

    best_by_stage = matrix.max(axis=0, skipna=True).to_dict()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue
        if col == -1:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        stage_key = ordered_stages[col]
        model_name = ordered_models[row - 1]
        raw_value = matrix.loc[model_name, stage_key]
        if pd.isna(raw_value):
            cell.set_facecolor(missing_cell_color)
            cell.set_text_props(color="#666666", style="italic")
            continue

        cell.set_facecolor(neutral_row if row % 2 == 1 else "white")
        if raw_value == best_by_stage.get(stage_key):
            cell.set_facecolor(best_cell_color)
            cell.set_text_props(color="white", weight="bold")

    ax.set_title("Best HOTA by Model and Stage", fontsize=13, pad=12)
    ax.text(
        0.5,
        -0.08,
        "Values rounded to 2 decimals. Best value within each stage column is highlighted. Missing stage entries are shown as —.",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_improvement_table_plot(
    improvement_df: pd.DataFrame,
    output_path: Path,
) -> None:
    stage_pair_order = [
        ("baseline", "finetuning", "Base -> FT"),
        ("baseline", "hyperparameter_tuning", "Base -> HPO"),
        ("baseline", "final_evaluation", "Base -> Final"),
        ("finetuning", "hyperparameter_tuning", "FT -> HPO"),
        ("hyperparameter_tuning", "final_evaluation", "HPO -> Final"),
    ]
    models = sorted(improvement_df["model"].dropna().unique())
    if not models:
        return

    display_rows: list[dict[str, Any]] = []
    for model in models:
        model_df = improvement_df.loc[
            (improvement_df["model"] == model)
            & (improvement_df["comparison_type"] == "best")
        ].copy()
        for from_stage, to_stage, label in stage_pair_order:
            match = model_df.loc[
                (model_df["from_stage"] == from_stage)
                & (model_df["to_stage"] == to_stage)
            ]
            if match.empty:
                display_rows.append(
                    {
                        "Model": model,
                        "Transition": label,
                        "Abs. Change": None,
                        "Rel. Change %": None,
                        "_relative_note": "Missing stage pair",
                        "_abs_state": "missing",
                    }
                )
                continue

            row = match.iloc[0]
            from_hota = float(row["from_HOTA"])
            to_hota = float(row["to_HOTA"])
            abs_change = float(row["absolute_improvement"])
            rel_change = row["relative_improvement_pct"]
            rel_display: float | None = None
            rel_note = None
            if pd.isna(rel_change):
                rel_note = "N/A (0 ref.)"
            else:
                rel_display = float(rel_change)

            if abs_change > 0:
                abs_state = "increase"
            elif abs_change < 0:
                abs_state = "decrease"
            else:
                abs_state = "no_change"

            display_rows.append(
                {
                    "Model": model,
                    "Transition": label,
                    "Abs. Change": abs_change,
                    "Rel. Change %": rel_display,
                    "_relative_note": rel_note,
                    "_abs_state": abs_state,
                }
            )

    display_df = pd.DataFrame(display_rows)
    plot_df = display_df.copy()
    plot_df["Abs. Change"] = plot_df["Abs. Change"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):+.2f}"
    )
    plot_df["Rel. Change %"] = plot_df.apply(
        lambda row: row["_relative_note"]
        if row["_relative_note"] is not None
        else f"{float(row['Rel. Change %']):+.1f}%",
        axis=1,
    )
    plot_df = plot_df[["Model", "Transition", "Abs. Change", "Rel. Change %"]]

    fig, ax = plt.subplots(
        figsize=(max(8.5, plot_df.shape[1] * 2.1), max(4.0, (len(plot_df) + 2) * 0.38))
    )
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    model_header_color = sns.color_palette("deep")[1]
    positive_color = "#DDEFD8"
    negative_color = "#F7D9D7"
    neutral_color = "#F6F6F6"
    block_fill = "#EAF2FB"

    table = ax.table(
        cellText=plot_df.to_numpy().tolist(),
        colLabels=list(plot_df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.25)

    previous_model = None
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        model = display_df.iloc[row - 1]["Model"]
        if model != previous_model:
            row_fill = block_fill
            previous_model = model
        else:
            row_fill = "white"

        if col in (0, 1):
            if col == 0 and (row == 1 or display_df.iloc[row - 2]["Model"] != model):
                cell.set_facecolor(model_header_color)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(row_fill)
                if col == 0:
                    cell.get_text().set_text("")
            if col == 1:
                cell.set_text_props(weight="bold")
            continue

        column_name = plot_df.columns[col]
        if column_name == "Abs. Change":
            state = display_df.iloc[row - 1]["_abs_state"]
            if state == "increase":
                cell.set_facecolor(positive_color)
                cell.set_text_props(weight="bold")
            elif state == "decrease":
                cell.set_facecolor(negative_color)
                cell.set_text_props(weight="bold")
            elif state == "no_change":
                cell.set_facecolor(neutral_color)
            else:
                cell.set_facecolor(neutral_color)
                cell.set_text_props(color="#666666")
            continue

        relative_note = display_df.iloc[row - 1]["_relative_note"]
        if isinstance(relative_note, str):
            cell.set_facecolor(neutral_color)
            cell.set_text_props(color="#666666", style="italic")
        else:
            cell.set_facecolor(row_fill)

    ax.set_title("HOTA improvement across stages", fontsize=13, pad=12)
    ax.text(
        0.5,
        -0.07,
        "Best-run stage deltas are shown. Green indicates improvement, red indicates decline.",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_decomposition_improvements_table_plot(
    decomposition_improvement_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if decomposition_improvement_df.empty:
        return

    comparison_labels = {
        "baseline -> finetuning": "Base -> FT",
        "baseline -> hyperparameter_tuning": "Base -> HPO",
        "baseline -> final_evaluation": "Base -> Final",
        "finetuning -> hyperparameter_tuning": "FT -> HPO",
        "hyperparameter_tuning -> final_evaluation": "HPO -> Final",
    }
    metric_columns = [
        ("delta_DetA", "DetA"),
        ("delta_AssA", "AssA"),
        ("delta_DetPr", "DetPr"),
        ("delta_DetRe", "DetRe"),
        ("delta_AssPr", "AssPr"),
        ("delta_AssRe", "AssRe"),
    ]

    display_rows: list[dict[str, Any]] = []
    models = sorted(decomposition_improvement_df["model"].dropna().unique())
    for model in models:
        model_df = decomposition_improvement_df.loc[
            decomposition_improvement_df["model"] == model
        ].copy()
        model_df["comparison"] = (
            model_df["from_stage"] + " -> " + model_df["to_stage"]
        )
        model_df["comparison_order"] = model_df["comparison"].map(
            {
                "baseline -> finetuning": 0,
                "baseline -> hyperparameter_tuning": 1,
                "baseline -> final_evaluation": 2,
                "finetuning -> hyperparameter_tuning": 3,
                "hyperparameter_tuning -> final_evaluation": 4,
            }
        )
        model_df = model_df.sort_values("comparison_order")
        for _, row in model_df.iterrows():
            display_row = {
                "Model": model,
                "Transition": comparison_labels.get(row["comparison"], row["comparison"]),
            }
            for source_col, display_col in metric_columns:
                value = row.get(source_col)
                display_row[display_col] = (
                    None if value is None or pd.isna(value) else float(value)
                )
            display_rows.append(display_row)

    display_df = pd.DataFrame(display_rows)
    if display_df.empty:
        return

    formatted_df = display_df.copy()
    for column in ["DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]:
        formatted_df[column] = formatted_df[column].map(
            lambda value: "" if value is None or pd.isna(value) else f"{float(value):.2f}"
        )

    fig, ax = plt.subplots(
        figsize=(max(10, formatted_df.shape[1] * 1.6), max(4, (len(formatted_df) + 2) * 0.38))
    )
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    model_header_color = sns.color_palette("deep")[1]
    positive_color = "#DDEFD8"
    negative_color = "#F7D9D7"
    neutral_color = "#F6F6F6"
    block_fill = "#EAF2FB"

    table = ax.table(
        cellText=formatted_df.to_numpy().tolist(),
        colLabels=list(formatted_df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.25)

    metric_start_cols = {"DetA": 2, "DetPr": 4, "AssPr": 6}
    previous_model = None
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        model = display_df.iloc[row - 1]["Model"]
        if model != previous_model:
            row_fill = block_fill
            previous_model = model
        else:
            row_fill = "white"

        if col in (0, 1):
            if col == 0 and (
                row == 1 or display_df.iloc[row - 2]["Model"] != model
            ):
                cell.set_facecolor(model_header_color)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(row_fill)
            if col == 1:
                cell.set_text_props(weight="bold")
            continue

        column_name = formatted_df.columns[col]
        raw_value = display_df.iloc[row - 1][column_name]
        if raw_value is None or pd.isna(raw_value):
            cell.set_facecolor(neutral_color)
            cell.set_text_props(color="#666666")
        elif raw_value > 0:
            cell.set_facecolor(positive_color)
        elif raw_value < 0:
            cell.set_facecolor(negative_color)
        else:
            cell.set_facecolor(neutral_color)

        if column_name in metric_start_cols:
            cell.set_linewidth(1.6)
            cell.set_edgecolor("#9BAFC7")

    ax.set_title(
        "Best-Run Decomposition Improvements Across Stages",
        fontsize=13,
        pad=12,
    )
    ax.text(
        0.5,
        -0.08,
        "Values rounded to 2 decimals. Green indicates improvement, red indicates decline.",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_decomposition_metrics_across_stages_table_plot(
    decomposition_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if decomposition_summary.empty:
        return

    stage_labels = {
        "baseline": "Base",
        "finetuning": "FT",
        "hyperparameter_tuning": "HPT",
        "final_evaluation": "Final",
    }
    metric_columns = [
        ("best_DetA", "DetA"),
        ("best_AssA", "AssA"),
        ("best_DetPr", "DetPr"),
        ("best_DetRe", "DetRe"),
        ("best_AssPr", "AssPr"),
        ("best_AssRe", "AssRe"),
    ]

    display_rows: list[dict[str, Any]] = []
    models = sorted(decomposition_summary["model"].dropna().unique())
    for model in models:
        model_df = decomposition_summary.loc[
            decomposition_summary["model"] == model
        ].copy()
        model_df["stage_order"] = model_df["stage"].map(STAGE_ORDER)
        model_df = model_df.sort_values("stage_order")
        for _, row in model_df.iterrows():
            display_row = {
                "Model": model,
                "Stage": stage_labels.get(row["stage"], row["stage"]),
            }
            for source_col, display_col in metric_columns:
                value = row.get(source_col)
                display_row[display_col] = (
                    None if value is None or pd.isna(value) else float(value)
                )
            display_rows.append(display_row)

    display_df = pd.DataFrame(display_rows)
    if display_df.empty:
        return

    formatted_df = display_df.copy()
    for column in ["DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]:
        formatted_df[column] = formatted_df[column].map(
            lambda value: "" if value is None or pd.isna(value) else f"{float(value):.2f}"
        )

    fig, ax = plt.subplots(
        figsize=(max(10, formatted_df.shape[1] * 1.6), max(4, (len(formatted_df) + 2) * 0.38))
    )
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    model_header_color = sns.color_palette("deep")[1]
    positive_color = "#DDEFD8"
    neutral_color = "#F6F6F6"
    block_fill = "#EAF2FB"

    table = ax.table(
        cellText=formatted_df.to_numpy().tolist(),
        colLabels=list(formatted_df.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.25)

    metric_start_cols = {"DetA": 2, "DetPr": 4, "AssPr": 6}
    previous_model = None
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        model = display_df.iloc[row - 1]["Model"]
        if model != previous_model:
            row_fill = block_fill
            previous_model = model
        else:
            row_fill = "white"

        if col in (0, 1):
            if col == 0 and (row == 1 or display_df.iloc[row - 2]["Model"] != model):
                cell.set_facecolor(model_header_color)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(row_fill)
                if col == 0:
                    cell.get_text().set_text("")
            if col == 1:
                cell.set_text_props(weight="bold")
            continue

        column_name = formatted_df.columns[col]
        raw_value = display_df.iloc[row - 1][column_name]
        if raw_value is None or pd.isna(raw_value):
            cell.set_facecolor(neutral_color)
            cell.set_text_props(color="#666666")
        elif float(raw_value) > 0:
            cell.set_facecolor(positive_color)
        else:
            cell.set_facecolor(neutral_color)

        if column_name in metric_start_cols:
            cell.set_linewidth(1.6)
            cell.set_edgecolor("#9BAFC7")

    ax.set_title(
        "Best-Run Decomposition Metrics Across Stages",
        fontsize=13,
        pad=12,
    )
    ax.text(
        0.5,
        -0.08,
        "Values rounded to 2 decimals. Best-run decomposition metrics are shown by model and stage.",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_simple_hpo_parameter_group_tests_table(
    hpo_parameter_group_tests: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if hpo_parameter_group_tests.empty:
        return

    data = hpo_parameter_group_tests.copy()
    if "ranking_score" not in data.columns:
        data["ranking_score"] = np.where(
            data["parameter_type"].eq("numeric"),
            data["effect_size"].abs(),
            data["group_spread_HOTA"],
        )
    selected_groups: list[pd.DataFrame] = []
    for model in sorted(data["model"].dropna().unique()):
        model_df = data.loc[data["model"] == model].copy()
        categorical = (
            model_df.loc[model_df["parameter_type"] == "categorical"]
            .sort_values(["ranking_score", "p_value"], ascending=[False, True], na_position="last")
            .head(2)
        )
        numeric = (
            model_df.loc[model_df["parameter_type"] == "numeric"]
            .sort_values(["ranking_score", "p_value"], ascending=[False, True], na_position="last")
            .head(3)
        )
        selected_groups.extend([categorical, numeric])
    data = pd.concat(selected_groups, ignore_index=True)
    if data.empty:
        return
    if not sensitivity_df.empty:
        interpretation_lookup = (
            sensitivity_df.loc[:, ["model", "display_parameter", "interpretation"]]
            .drop_duplicates()
        )
        data = data.merge(
            interpretation_lookup,
            on=["model", "display_parameter"],
            how="left",
        )
    else:
        data["interpretation"] = None

    def fmt_p(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        value = float(value)
        if value < 0.001:
            return f"{value:.1e}"
        return f"{value:.3f}"

    def build_direction_or_group(row: pd.Series) -> str:
        if row["parameter_type"] == "numeric":
            direction = "positive" if float(row["effect_size"]) > 0 else "negative"
            return f"{direction} association"
        best_group = row.get("best_group_label")
        best_hota = row.get("best_group_median_HOTA")
        if pd.notna(best_group) and pd.notna(best_hota):
            return f"best: {best_group} ({float(best_hota):.3f})"
        return ""

    def build_basis(row: pd.Series) -> str:
        if row["parameter_type"] == "numeric":
            return r"$|\rho_s|$ with HOTA"
        return "Median HOTA spread"

    display_rows: list[dict[str, Any]] = []
    for model in sorted(data["model"].dropna().unique()):
        model_df = data.loc[data["model"] == model].copy()
        categorical_df = model_df.loc[model_df["parameter_type"] == "categorical"]
        numeric_df = model_df.loc[model_df["parameter_type"] == "numeric"]

        if not categorical_df.empty:
            display_rows.append(
                {
                    "Model": model,
                    "Parameter": "Categorical parameters",
                    "Importance": "",
                    "Basis": "",
                    "Practical meaning": "",
                    "p-value": "",
                    "Interpretation": "",
                    "_parameter_type": "section_categorical",
                    "_is_supported": True,
                    "_model": model,
                }
            )
            for _, row in categorical_df.iterrows():
                display_rows.append(
                    {
                        "Model": "",
                        "Parameter": row["display_parameter"],
                        "Importance": f"{float(row['ranking_score']):.3f}" if pd.notna(row["ranking_score"]) else "",
                        "Basis": "Median spread",
                        "Practical meaning": build_direction_or_group(row),
                        "p-value": fmt_p(row["p_value"]),
                        "Interpretation": row.get("interpretation") or "",
                        "_parameter_type": row["parameter_type"],
                        "_is_supported": bool(pd.notna(row["p_value"]) and float(row["p_value"]) < 0.05),
                        "_model": model,
                    }
                )

        if not numeric_df.empty:
            display_rows.append(
                {
                    "Model": "" if not categorical_df.empty else model,
                    "Parameter": "Numeric parameters",
                    "Importance": "",
                    "Basis": "",
                    "Practical meaning": "",
                    "p-value": "",
                    "Interpretation": "",
                    "_parameter_type": "section_numeric",
                    "_is_supported": True,
                    "_model": model,
                }
            )
            for _, row in numeric_df.iterrows():
                display_rows.append(
                    {
                        "Model": "",
                        "Parameter": row["display_parameter"],
                        "Importance": f"{float(row['ranking_score']):.3f}" if pd.notna(row["ranking_score"]) else "",
                        "Basis": "Abs. Spearman",
                        "Practical meaning": build_direction_or_group(row),
                        "p-value": fmt_p(row["p_value"]),
                        "Interpretation": row.get("interpretation") or "",
                        "_parameter_type": row["parameter_type"],
                        "_is_supported": bool(pd.notna(row["p_value"]) and float(row["p_value"]) < 0.05),
                        "_model": model,
                    }
                )

    plot_df = pd.DataFrame(display_rows)
    fig, ax = plt.subplots(figsize=(16.5, max(4.8, len(plot_df) * 0.44)))
    ax.axis("off")
    header_color = sns.color_palette("deep")[0]
    type_numeric_fill = "#EAF2FB"
    type_categorical_fill = "#F7F0E3"
    section_numeric_fill = "#DCEAF8"
    section_categorical_fill = "#F3E6D2"
    regular_fill = "white"

    table = ax.table(
        cellText=plot_df[
            [
                "Model",
                "Parameter",
                "Importance",
                "Basis",
                "Practical meaning",
                "p-value",
                "Interpretation",
            ]
        ].to_numpy().tolist(),
        colLabels=[
            "Model",
            "Parameter",
            "Importance",
            "Basis",
            "Practical meaning",
            "p-value",
            "Interpretation",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.1, 0.2, 0.1, 0.12, 0.19, 0.07, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 1.22)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue

        data_row = plot_df.iloc[row - 1]
        cell.set_facecolor(regular_fill)

        if data_row["_parameter_type"] == "section_categorical":
            cell.set_facecolor(section_categorical_fill)
            if col == 1:
                cell.set_text_props(weight="bold")
            if col == 0 and data_row["Model"]:
                cell.set_text_props(weight="bold")
            continue
        if data_row["_parameter_type"] == "section_numeric":
            cell.set_facecolor(section_numeric_fill)
            if col == 1:
                cell.set_text_props(weight="bold")
            continue

        if col == 2:
            cell.set_facecolor(regular_fill)
        if col == 3:
            cell.set_facecolor(
                type_numeric_fill
                if data_row["_parameter_type"] == "numeric"
                else type_categorical_fill
            )
        if not data_row["_is_supported"] and col in [1, 2, 3, 4, 5, 6]:
            cell.set_text_props(color="#666666")

    previous_model = None
    for row_idx, row in plot_df.iterrows():
        model = row["_model"]
        table_row = row_idx + 1
        if previous_model is not None and model != previous_model:
            for col_idx in range(7):
                table[(table_row, col_idx)].set_linewidth(1.6)
                table[(table_row, col_idx)].set_edgecolor("#A0A0A0")
        previous_model = model

    ax.set_title("Hyperparameter Importance on HOTA by Model", fontsize=13, pad=12)
    ax.text(
        0.5,
        -0.07,
        "Rows are grouped by model, with separate categorical and numeric sections. Importance is based on median HOTA spread for categorical parameters and $|\\rho_s|$ for numeric parameters.",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_table_style_plots(artifacts: PipelineArtifacts, plots_dir: Path) -> None:
    descriptive_df = artifacts.descriptive_summary.copy()
    improvement_df = artifacts.improvement_summary.copy()
    raw_df = artifacts.raw_runs.copy()
    cleaned_df = artifacts.cleaned_runs.copy()
    sensitivity_df = artifacts.hyperparameter_sensitivity.copy()

    render_best_hota_stage_matrix_plot(
        descriptive_df,
        plots_dir / "best_hota_by_stage_table.png",
    )

    validation_vs_final_table = descriptive_df.loc[
        descriptive_df["stage"].isin(["hyperparameter_tuning", "final_evaluation"]),
        ["model", "stage", "max_HOTA"],
    ].dropna(subset=["max_HOTA"])
    render_table_plot(
        validation_vs_final_table,
        "Best Validation HOTA vs Final Test HOTA",
        plots_dir / "validation_best_vs_final_test_table.png",
    )

    improvement_table = improvement_df.loc[
        improvement_df["comparison_type"] == "best"
    ].copy()
    render_improvement_table_plot(
        improvement_table,
        plots_dir / "hota_improvement_across_stages_table.png",
    )

    render_decomposition_improvements_table_plot(
        artifacts.decomposition_improvement_summary.copy(),
        plots_dir / "decomposition_improvements_across_stages_table.png",
    )
    render_decomposition_metrics_across_stages_table_plot(
        artifacts.decomposition_summary.copy(),
        plots_dir / "decomposition_metrics_across_stages_table.png",
    )

    hpo_stage_table = artifacts.hpo_stage_summary.loc[
        :,
        [
            "model",
            "median_HOTA",
            "best_HOTA",
            "top_quartile_mean_HOTA",
            "proportion_above_baseline",
            "proportion_within_5pct_of_best",
        ],
    ].copy()
    render_table_plot(
        hpo_stage_table,
        "HPO Stage Summary Overview",
        plots_dir / "hpo_stage_summary_overview_table.png",
    )

    hpo_convergence_table = artifacts.hpo_convergence_summary.loc[
        :,
        [
            "model",
            "first_trial_reaching_90pct_of_best",
            "first_trial_reaching_95pct_of_best",
            "best_so_far_gain_over_first_trial",
            "improvement_last_minus_first_trial",
        ],
    ].copy()
    render_table_plot(
        hpo_convergence_table,
        "HPO Convergence Summary Overview",
        plots_dir / "hpo_convergence_summary_overview_table.png",
    )

    hpo_model_comparison_table = artifacts.hpo_model_comparison.loc[
        :,
        [
            "model",
            "best_HOTA",
            "median_HOTA",
            "top_decile_mean_HOTA",
            "iqr_HOTA",
            "first_trial_reaching_95pct_of_best",
            "is_best_hpo_model",
            "is_most_robust_hpo_model",
            "is_most_hpo_efficient_model",
        ],
    ].copy()
    render_table_plot(
        hpo_model_comparison_table,
        "HPO Model Comparison Overview",
        plots_dir / "hpo_model_comparison_overview_table.png",
    )

    hpo_parameter_table = artifacts.hpo_parameter_group_tests.copy()
    render_simple_hpo_parameter_group_tests_table(
        hpo_parameter_table,
        artifacts.hyperparameter_sensitivity.copy(),
        plots_dir / "hpo_parameter_group_tests_overview_table.png",
    )

    motip_area_thresh_table = build_grouped_parameter_table(
        cleaned_df, raw_df, sensitivity_df, "MOTIP", "AREA THRESH"
    )
    render_table_plot(
        motip_area_thresh_table,
        "MOTIP AREA THRESH Grouped HOTA",
        plots_dir / "motip_area_thresh_grouped_table.png",
    )

    motip_assignment_protocol_table = build_grouped_parameter_table(
        cleaned_df, raw_df, sensitivity_df, "MOTIP", "assignment protocol"
    )
    render_table_plot(
        motip_assignment_protocol_table,
        "MOTIP Assignment Protocol Grouped HOTA",
        plots_dir / "motip_assignment_protocol_grouped_table.png",
    )


def annotate_bar_values(
    ax: Any,
    fmt: str = "{:.2f}",
    orientation: str = "vertical",
    padding_fraction: float = 0.02,
) -> None:
    if orientation == "vertical":
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * padding_fraction if ymax > ymin else 0.05
        for patch in ax.patches:
            value = patch.get_height()
            if np.isnan(value):
                continue
            x = patch.get_x() + patch.get_width() / 2
            if value >= 0:
                y = value + pad
                va = "bottom"
            else:
                y = value - pad
                va = "top"
            ax.text(x, y, fmt.format(value), ha="center", va=va, fontsize=8)
    else:
        xmin, xmax = ax.get_xlim()
        pad = (xmax - xmin) * padding_fraction if xmax > xmin else 0.05
        for patch in ax.patches:
            value = patch.get_width()
            if np.isnan(value):
                continue
            y = patch.get_y() + patch.get_height() / 2
            if value >= 0:
                x = value + pad
                ha = "left"
            else:
                x = value - pad
                ha = "right"
            ax.text(x, y, fmt.format(value), ha=ha, va="center", fontsize=8)


def annotate_hue_medians(
    ax: Any,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    x_order: list[Any],
    hue_order: list[Any],
    fmt: str = "{:.2f}",
) -> None:
    if not x_order or not hue_order:
        return
    n_hues = len(hue_order)
    width = 0.8
    grouped = (
        data.groupby([x_col, hue_col], observed=False)[y_col]
        .median()
        .reset_index()
        .dropna(subset=[y_col])
    )
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.01 if y_max > y_min else 0.05
    for _, row in grouped.iterrows():
        if row[x_col] not in x_order or row[hue_col] not in hue_order:
            continue
        x_index = x_order.index(row[x_col])
        hue_index = hue_order.index(row[hue_col])
        x_position = x_index - width / 2 + (hue_index + 0.5) * (width / n_hues)
        ax.text(
            x_position,
            float(row[y_col]) + pad,
            fmt.format(float(row[y_col])),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
        )


def plot_hota_distributions(cleaned_df: pd.DataFrame, plots_dir: Path) -> None:
    if cleaned_df.empty:
        return
    data = cleaned_df.copy()
    data["HOTA"] = maybe_numeric(data["HOTA"])
    model_order = sorted(data["model"].dropna().unique())
    if not model_order:
        return

    single_stages = ["baseline", "finetuning", "final_evaluation"]
    stage_labels = {
        "baseline": "Baseline",
        "finetuning": "Finetuning",
        "final_evaluation": "Final Eval.",
        "hyperparameter_tuning": "HPO",
    }
    palette = {
        "baseline": "#C9D3DD",
        "finetuning": "#4C72B0",
        "final_evaluation": "#55A868",
        "hyperparameter_tuning": "#DD8452",
    }

    fig, axes = plt.subplots(
        nrows=len(model_order),
        ncols=2,
        figsize=(14, max(4.5, 3.8 * len(model_order))),
        gridspec_kw={"width_ratios": [1.2, 1.8]},
        sharey=False,
    )
    if len(model_order) == 1:
        axes = np.array([axes])

    for row_index, model in enumerate(model_order):
        model_df = data.loc[data["model"] == model].copy()
        left_ax, right_ax = axes[row_index]

        single_df = (
            model_df.loc[model_df["stage"].isin(single_stages), ["stage", "HOTA"]]
            .dropna()
            .copy()
        )
        present_single_stages = [
            stage for stage in single_stages if stage in single_df["stage"].tolist()
        ]
        x_positions = {stage: idx for idx, stage in enumerate(single_stages)}

        if not single_df.empty:
            for stage in present_single_stages:
                stage_df = single_df.loc[single_df["stage"] == stage]
                value = float(stage_df["HOTA"].iloc[0])
                x = x_positions[stage]
                left_ax.scatter(
                    [x],
                    [value],
                    s=80,
                    color=palette[stage],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
                left_ax.text(
                    x,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
                )
                left_ax.text(
                    x,
                    0.03,
                    "n=1",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    transform=left_ax.get_xaxis_transform(),
                )
            left_ax.set_xticks(range(len(single_stages)))
            left_ax.set_xticklabels([stage_labels[stage] for stage in single_stages])
        else:
            left_ax.set_xticks(range(len(single_stages)))
            left_ax.set_xticklabels([stage_labels[stage] for stage in single_stages])

        left_ax.set_title(f"{model}: Single-Run Reference Stages")
        left_ax.set_xlabel("")
        left_ax.set_ylabel("HOTA")
        left_ax.grid(axis="y", alpha=0.25)

        hpo_df = (
            model_df.loc[model_df["stage"] == "hyperparameter_tuning", ["HOTA"]]
            .dropna()
            .copy()
        )
        if not hpo_df.empty:
            hpo_df["stage_label"] = "HPO"
            sns.violinplot(
                data=hpo_df,
                x="stage_label",
                y="HOTA",
                inner=None,
                cut=0,
                color=palette["hyperparameter_tuning"],
                ax=right_ax,
            )
            sns.stripplot(
                data=hpo_df,
                x="stage_label",
                y="HOTA",
                color="black",
                alpha=0.45,
                size=3.2,
                ax=right_ax,
            )
            median_value = float(hpo_df["HOTA"].median())
            right_ax.axhline(
                median_value,
                color="#8C2D04",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
            right_ax.text(
                0,
                median_value,
                f"median={median_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            )
            right_ax.text(
                0,
                0.03,
                f"n={len(hpo_df)}",
                ha="center",
                va="bottom",
                fontsize=8,
                transform=right_ax.get_xaxis_transform(),
            )
        else:
            right_ax.text(
                0.5,
                0.5,
                "No HPO distribution available",
                ha="center",
                va="center",
                transform=right_ax.transAxes,
                fontsize=9,
            )
            right_ax.set_xticks([0])
            right_ax.set_xticklabels(["HPO"])

        right_ax.set_title(f"{model}: Hyperparameter Tuning Distribution")
        right_ax.set_xlabel("")
        right_ax.set_ylabel("HOTA")
        right_ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Single-Run Stage Values and HPO Distributions by Model", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.01,
        "Single-run stages are shown as labeled points. HPO is shown as a violin distribution with trial-level points and a labeled median.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plots_dir / "hota_distributions_by_model_stage.png", dpi=200)
    plt.close()


def plot_best_hota(descriptive_df: pd.DataFrame, plots_dir: Path) -> None:
    data = descriptive_df.dropna(subset=["max_HOTA"]).copy()
    if data.empty:
        return
    data["stage"] = pd.Categorical(
        data["stage"], categories=NORMALIZED_STAGES, ordered=True
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="stage", y="max_HOTA", hue="model")
    plt.title("Best HOTA by Model and Stage")
    plt.ylabel("Best HOTA")
    plt.xlabel("Stage")
    annotate_bar_values(ax)
    plt.tight_layout()
    plt.savefig(plots_dir / "best_hota_by_stage.png", dpi=200)
    plt.close()


def plot_improvements(improvement_df: pd.DataFrame, plots_dir: Path) -> None:
    data = improvement_df.loc[improvement_df["comparison_type"] == "best"].copy()
    if data.empty:
        return
    labels = data["from_stage"] + " -> " + data["to_stage"]
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=data.assign(comparison=labels),
        x="comparison",
        y="absolute_improvement",
        hue="model",
    )
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Best-Run HOTA Improvements Across Stages")
    plt.ylabel("Absolute HOTA Improvement")
    plt.xlabel("Stage Comparison")
    plt.xticks(rotation=20, ha="right")
    annotate_bar_values(ax)
    plt.tight_layout()
    plt.savefig(plots_dir / "hota_improvement_across_stages.png", dpi=200)
    plt.close()


def plot_validation_vs_final(descriptive_df: pd.DataFrame, plots_dir: Path) -> None:
    data = descriptive_df.loc[
        descriptive_df["stage"].isin(["hyperparameter_tuning", "final_evaluation"]),
        ["model", "stage", "max_HOTA"],
    ].dropna()
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="model", y="max_HOTA", hue="stage")
    plt.title("Best Validation HOTA vs Final Test HOTA")
    plt.ylabel("Best HOTA")
    plt.xlabel("Model")
    annotate_bar_values(ax)
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_best_vs_final_test.png", dpi=200)
    plt.close()


def plot_deta_vs_assa(
    decomposition_summary: pd.DataFrame, plots_dir: Path
) -> None:
    if decomposition_summary.empty:
        return
    data = decomposition_summary.dropna(subset=["best_DetA", "best_AssA"]).copy()
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="best_DetA", y="best_AssA", hue="model", style="stage", s=90)
    for _, row in data.iterrows():
        plt.text(
            float(row["best_DetA"]),
            float(row["best_AssA"]),
            f"{row['model']}:{row['stage']}",
            fontsize=7,
            alpha=0.8,
        )
    plt.title("Best-Run DetA vs AssA by Model and Stage")
    plt.xlabel("Best-run DetA")
    plt.ylabel("Best-run AssA")
    plt.tight_layout()
    plt.savefig(plots_dir / "deta_vs_assa_by_model_stage.png", dpi=200)
    plt.close()


def plot_decomposition_improvements(
    decomposition_improvement_summary: pd.DataFrame, plots_dir: Path
) -> None:
    if decomposition_improvement_summary.empty:
        return
    data = decomposition_improvement_summary.copy()
    data["comparison"] = data["from_stage"] + " -> " + data["to_stage"]
    long_df = data.melt(
        id_vars=["model", "comparison"],
        value_vars=[
            "delta_DetA",
            "delta_AssA",
            "delta_DetPr",
            "delta_DetRe",
            "delta_AssPr",
            "delta_AssRe",
        ],
        var_name="metric",
        value_name="delta",
    ).dropna(subset=["delta"])
    if long_df.empty:
        return
    long_df["metric"] = long_df["metric"].str.removeprefix("delta_")
    metric_order = ["DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]
    comparison_order = [
        "baseline -> finetuning",
        "baseline -> hyperparameter_tuning",
        "baseline -> final_evaluation",
        "finetuning -> hyperparameter_tuning",
        "hyperparameter_tuning -> final_evaluation",
    ]
    models = list(long_df["model"].dropna().unique())
    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        figsize=(14, max(4, 4.2 * len(models))),
        sharex=False,
        sharey=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    palette = sns.color_palette("tab10", n_colors=len(metric_order))
    for ax, model in zip(axes, models):
        model_df = long_df.loc[long_df["model"] == model].copy()
        sns.barplot(
            data=model_df,
            x="comparison",
            y="delta",
            hue="metric",
            order=[c for c in comparison_order if c in model_df["comparison"].unique()],
            hue_order=metric_order,
            palette=palette,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"{model}: Best-Run Decomposition Changes Across Stages")
        ax.set_xlabel("Stage Comparison")
        ax.set_ylabel("Final stage - previous stage")
        ax.tick_params(axis="x", rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        annotate_bar_values(ax, padding_fraction=0.01)
        if ax is not axes[0]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    first_legend = axes[0].get_legend()
    if first_legend is not None:
        first_legend.set_title("Metric")
        first_legend.set_bbox_to_anchor((1.02, 1.0))
        first_legend._loc = 2
    fig.suptitle(
        "Best-Run Decomposition Metric Improvements Across Stages",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 0.88, 0.98))
    plt.savefig(plots_dir / "decomposition_improvements_across_stages.png", dpi=200)
    plt.close()


def plot_decomposition_metrics_across_stages(
    decomposition_summary: pd.DataFrame, plots_dir: Path
) -> None:
    if decomposition_summary.empty:
        return
    data = decomposition_summary.copy()
    metric_columns = [
        "best_DetA",
        "best_AssA",
        "best_DetPr",
        "best_DetRe",
        "best_AssPr",
        "best_AssRe",
    ]
    long_df = data.melt(
        id_vars=["model", "stage"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return
    long_df["metric"] = long_df["metric"].str.removeprefix("best_")
    stage_order = [stage for stage in NORMALIZED_STAGES if stage in long_df["stage"].unique()]
    stage_labels = {
        "baseline": "Baseline",
        "finetuning": "Finetuning",
        "hyperparameter_tuning": "HPT",
        "final_evaluation": "Final",
    }
    metric_order = ["DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe"]
    models = list(long_df["model"].dropna().unique())
    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        figsize=(14, max(4, 4.2 * len(models))),
        sharex=False,
        sharey=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    palette = sns.color_palette("tab10", n_colors=len(metric_order))
    for ax, model in zip(axes, models):
        model_df = long_df.loc[long_df["model"] == model].copy()
        sns.barplot(
            data=model_df,
            x="stage",
            y="value",
            hue="metric",
            order=stage_order,
            hue_order=metric_order,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{model}: Best-Run Decomposition Metrics by Stage")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Best-run metric value")
        ax.set_xticklabels([stage_labels.get(stage, stage) for stage in stage_order])
        annotate_bar_values(ax, padding_fraction=0.01)
        if ax is not axes[0]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    first_legend = axes[0].get_legend()
    if first_legend is not None:
        first_legend.set_title("Metric")
        first_legend.set_bbox_to_anchor((1.02, 1.0))
        first_legend._loc = 2
    fig.suptitle(
        "Best-Run Decomposition Metrics Across Stages",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 0.88, 0.98))
    plt.savefig(plots_dir / "decomposition_metrics_across_stages.png", dpi=200)
    plt.close()


def plot_best_validation_to_final_decomposition(
    best_validation_to_final_decomposition: pd.DataFrame, plots_dir: Path
) -> None:
    if best_validation_to_final_decomposition.empty:
        return
    data = best_validation_to_final_decomposition.copy()
    long_df = data.melt(
        id_vars=["model"],
        value_vars=["delta_HOTA", "delta_DetA", "delta_AssA"],
        var_name="metric",
        value_name="delta",
    ).dropna(subset=["delta"])
    if long_df.empty:
        return
    long_df["metric"] = long_df["metric"].str.removeprefix("delta_")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=long_df, x="model", y="delta", hue="metric")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Best HPO Validation Trial to Final Test: HOTA, DetA, and AssA Changes")
    plt.xlabel("Model")
    plt.ylabel("Final test - best HPO validation")
    annotate_bar_values(ax)
    plt.tight_layout()
    plt.savefig(plots_dir / "best_validation_to_final_decomposition.png", dpi=200)
    plt.close()


def plot_precision_recall_profiles(
    decomposition_summary: pd.DataFrame,
    precision_col: str,
    recall_col: str,
    title: str,
    filename: str,
    plots_dir: Path,
) -> None:
    if decomposition_summary.empty:
        return
    data = decomposition_summary.dropna(subset=[precision_col, recall_col]).copy()
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x=precision_col,
        y=recall_col,
        hue="model",
        style="stage",
        s=90,
    )
    for _, row in data.iterrows():
        plt.text(
            float(row[precision_col]),
            float(row[recall_col]),
            f"{row['model']}:{row['stage']}",
            fontsize=7,
            alpha=0.8,
        )
    plt.title(title)
    plt.xlabel(precision_col.replace("best_", "").replace("mean_", ""))
    plt.ylabel(recall_col.replace("best_", "").replace("mean_", ""))
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=200)
    plt.close()


def plot_hpo_trial_progression(cleaned_df: pd.DataFrame, plots_dir: Path) -> None:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    if hpo_df.empty:
        return
    plot_df = hpo_df.sort_values(["model", "trial_order"]).copy()
    plot_df["best_so_far_HOTA"] = plot_df.groupby("model")["HOTA"].cummax()
    fig, axes = plt.subplots(
        nrows=len(plot_df["model"].dropna().unique()),
        ncols=1,
        figsize=(10, max(4, 3 * len(plot_df["model"].dropna().unique()))),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (model, group) in zip(axes, plot_df.groupby("model", sort=False)):
        sns.scatterplot(data=group, x="trial_order", y="HOTA", ax=ax, s=35)
        sns.lineplot(data=group, x="trial_order", y="best_so_far_HOTA", ax=ax, color="orange")
        ax.set_title(f"{model}: HPO trial progression")
        ax.set_xlabel("Trial order")
        ax.set_ylabel("HOTA")
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_trial_progression_by_model.png", dpi=200)
    plt.close()


def plot_hpo_ecdf(cleaned_df: pd.DataFrame, plots_dir: Path) -> None:
    hpo_df = get_hpo_analysis_df(cleaned_df)
    if hpo_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=hpo_df, x="HOTA", hue="model")
    plt.title("HPO HOTA ECDF by Model")
    plt.xlabel("HOTA")
    plt.ylabel("Empirical CDF")
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_hota_ecdf_by_model.png", dpi=200)
    plt.close()


def plot_hpo_stage_summary_overview(
    hpo_stage_summary: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_stage_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary = hpo_stage_summary.sort_values("model").copy()
    long_hota = summary.melt(
        id_vars="model",
        value_vars=["median_HOTA", "best_HOTA", "top_quartile_mean_HOTA"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=long_hota, x="model", y="value", hue="metric", ax=axes[0])
    axes[0].set_title("HPO Stage Summary: Performance Levels")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("HOTA")
    prop_long = summary.melt(
        id_vars="model",
        value_vars=["proportion_above_baseline", "proportion_within_5pct_of_best"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    sns.barplot(data=prop_long, x="model", y="value", hue="metric", ax=axes[1])
    axes[1].set_title("HPO Stage Summary: Robustness Proportions")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Proportion")
    axes[1].set_ylim(0, 1.05)
    annotate_bar_values(axes[0])
    annotate_bar_values(axes[1], fmt="{:.0%}", padding_fraction=0.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_stage_summary_overview.png", dpi=200)
    plt.close()


def plot_hpo_convergence_summary_overview(
    hpo_convergence_summary: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_convergence_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary = hpo_convergence_summary.sort_values("model").copy()
    trial_long = summary.melt(
        id_vars="model",
        value_vars=[
            "first_trial_reaching_90pct_of_best",
            "first_trial_reaching_95pct_of_best",
        ],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=trial_long, x="model", y="value", hue="metric", ax=axes[0])
    axes[0].set_title("HPO Convergence: Trial Budget Needed")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Trial order")
    gain_long = summary.melt(
        id_vars="model",
        value_vars=[
            "best_so_far_gain_over_first_trial",
            "improvement_last_minus_first_trial",
        ],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=gain_long, x="model", y="value", hue="metric", ax=axes[1])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("HPO Convergence: Search Gains")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("HOTA difference")
    annotate_bar_values(axes[0], fmt="{:.0f}")
    annotate_bar_values(axes[1])
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_convergence_summary_overview.png", dpi=200)
    plt.close()


def plot_hpo_model_comparison_overview(
    hpo_model_comparison: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_model_comparison.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    comparison = hpo_model_comparison.sort_values("model").copy()
    perf_long = comparison.melt(
        id_vars="model",
        value_vars=["best_HOTA", "median_HOTA", "top_decile_mean_HOTA"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=perf_long, x="model", y="value", hue="metric", ax=axes[0])
    axes[0].set_title("HPO Model Comparison: Performance")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("HOTA")
    eff_long = comparison.melt(
        id_vars="model",
        value_vars=["iqr_HOTA", "first_trial_reaching_95pct_of_best"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=eff_long, x="model", y="value", hue="metric", ax=axes[1])
    axes[1].set_title("HPO Model Comparison: Robustness and Efficiency")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Value")
    annotate_bar_values(axes[0])
    annotate_bar_values(axes[1])
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_model_comparison_overview.png", dpi=200)
    plt.close()


def plot_hpo_statistical_analysis_overview(
    hpo_statistical_analysis: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_statistical_analysis.empty:
        return
    data = hpo_statistical_analysis.loc[
        hpo_statistical_analysis["analysis_name"].isin(
            ["hpo_vs_baseline", "hpo_vs_finetuning", "top_quartile_vs_rest", "late_vs_early_trials"]
        )
    ].copy()
    if data.empty:
        return
    data["label"] = data["model"] + ": " + data["analysis_name"].str.replace("_", " ")
    plt.figure(figsize=(12, max(5, 0.5 * len(data))))
    sns.pointplot(data=data, y="label", x="mean_difference_vs_reference", join=False)
    for _, row in data.iterrows():
        if pd.notna(row["mean_difference_ci95_low"]) and pd.notna(row["mean_difference_ci95_high"]):
            plt.plot(
                [row["mean_difference_ci95_low"], row["mean_difference_ci95_high"]],
                [row["label"], row["label"]],
                color="black",
                linewidth=1.5,
            )
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title("HPO Statistical Analysis: Mean Differences with 95% CIs")
    plt.xlabel("Mean difference vs reference")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_statistical_analysis_overview.png", dpi=200)
    plt.close()


def plot_hpo_parameter_group_tests_overview(
    hpo_parameter_group_tests: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_parameter_group_tests.empty:
        return
    data = hpo_parameter_group_tests.copy()
    if "ranking_score" not in data.columns:
        data["ranking_score"] = np.where(
            data["parameter_type"].eq("numeric"),
            data["effect_size"].abs(),
            data["group_spread_HOTA"],
        )
    selected_groups: list[pd.DataFrame] = []
    for model in sorted(data["model"].dropna().unique()):
        model_df = data.loc[data["model"] == model].copy()
        categorical = (
            model_df.loc[model_df["parameter_type"] == "categorical"]
            .sort_values(["ranking_score", "p_value"], ascending=[False, True], na_position="last")
            .head(2)
        )
        numeric = (
            model_df.loc[model_df["parameter_type"] == "numeric"]
            .sort_values(["ranking_score", "p_value"], ascending=[False, True], na_position="last")
            .head(3)
        )
        selected_groups.extend([categorical, numeric])
    data = pd.concat(selected_groups, ignore_index=True).copy()
    if data.empty:
        return
    data["label"] = data["model"] + ": " + data["display_parameter"]
    plt.figure(figsize=(12, max(5, 0.45 * len(data))))
    ax = sns.barplot(
        data=data,
        y="label",
        x="ranking_score",
        hue="parameter_type",
        dodge=False,
    )
    plt.title("HPO Hyperparameter Importance by Model")
    plt.xlabel("Thesis-oriented importance score")
    plt.ylabel("")
    annotate_bar_values(ax, orientation="horizontal")
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_parameter_group_tests_overview.png", dpi=200)
    plt.close()


def plot_hpo_cross_model_tests_overview(
    hpo_cross_model_tests: pd.DataFrame, plots_dir: Path
) -> None:
    if hpo_cross_model_tests.empty:
        return
    data = hpo_cross_model_tests.loc[
        hpo_cross_model_tests["comparison_type"] == "pairwise"
    ].copy()
    if data.empty:
        return
    data["label"] = data["model_b"] + " - " + data["model_a"]
    plt.figure(figsize=(10, max(4, 0.8 * len(data))))
    sns.pointplot(data=data, y="label", x="median_difference_b_minus_a", join=False)
    for _, row in data.iterrows():
        if pd.notna(row["bootstrap_median_diff_ci95_low"]) and pd.notna(row["bootstrap_median_diff_ci95_high"]):
            plt.plot(
                [row["bootstrap_median_diff_ci95_low"], row["bootstrap_median_diff_ci95_high"]],
                [row["label"], row["label"]],
                color="black",
                linewidth=1.5,
            )
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title("HPO Cross-Model Tests: Pairwise Median Differences")
    plt.xlabel("Median difference (model_b - model_a)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(plots_dir / "hpo_cross_model_tests_overview.png", dpi=200)
    plt.close()


def plot_hyperparameter_sensitivity(
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    if sensitivity_df.empty:
        return

    for model in sorted(sensitivity_df["model"].dropna().unique()):
        model_rows = sensitivity_df.loc[sensitivity_df["model"] == model].head(3)
        hpo_ids = set(
            cleaned_df.loc[
                (cleaned_df["model"] == model)
                & (cleaned_df["stage"] == "hyperparameter_tuning"),
                "run_id",
            ]
        )
        model_raw = raw_df.loc[raw_df["run_id"].isin(hpo_ids)].copy()
        if model_raw.empty:
            continue

        numeric_heatmap_values: dict[str, float] = {}
        for _, row in model_rows.iterrows():
            column = row["source_column"]
            parameter = row.get("display_parameter") or row["parameter"]
            if row["parameter_type"] == "numeric":
                data = pd.DataFrame(
                    {
                        "param": maybe_numeric(model_raw[column]),
                        "HOTA": maybe_numeric(model_raw["normalized_HOTA"]),
                    }
                ).dropna()
                if data.empty:
                    continue
                numeric_heatmap_values[parameter] = float(row["spearman_rho"])
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=data, x="param", y="HOTA")
                if len(data) >= 2:
                    sns.regplot(
                        data=data,
                        x="param",
                        y="HOTA",
                        scatter=False,
                        ci=None,
                        color="orange",
                    )
                plt.title(f"{model}: {parameter} vs HOTA")
                plt.xlabel(parameter)
                plt.tight_layout()
                plt.savefig(
                    plots_dir / f"{slugify(model)}_{slugify(parameter)}_scatter.png",
                    dpi=200,
                )
                plt.close()
            else:
                data = pd.DataFrame(
                    {
                        "param": model_raw[column].astype(str),
                        "HOTA": maybe_numeric(model_raw["normalized_HOTA"]),
                    }
                ).dropna()
                if data.empty:
                    continue
                grouped = (
                    data.groupby("param")
                    .agg(mean_HOTA=("HOTA", "mean"), count=("HOTA", "count"))
                    .reset_index()
                    .sort_values("mean_HOTA", ascending=False)
                    .head(8)
                )
                plt.figure(figsize=(7, 4))
                ax = sns.barplot(data=grouped, x="param", y="mean_HOTA")
                plt.title(f"{model}: {parameter} grouped HOTA")
                plt.xlabel(parameter)
                plt.ylabel("Mean HOTA")
                plt.xticks(rotation=20, ha="right")
                annotate_bar_values(ax)
                plt.tight_layout()
                plt.savefig(
                    plots_dir / f"{slugify(model)}_{slugify(parameter)}_grouped.png",
                    dpi=200,
                )
                plt.close()

        if numeric_heatmap_values:
            heatmap_df = pd.DataFrame.from_dict(
                numeric_heatmap_values, orient="index", columns=["Spearman rho"]
            )
            plt.figure(figsize=(6, max(2, 0.5 * len(heatmap_df))))
            sns.heatmap(
                heatmap_df, annot=True, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0
            )
            plt.title(f"{model}: Numeric Hyperparameter Correlations")
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"{slugify(model)}_hyperparameter_correlation_heatmap.png",
                dpi=200,
            )
            plt.close()


def generate_all_plots(
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    decomposition_improvement_summary: pd.DataFrame,
    best_validation_to_final_decomposition: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    hpo_stage_summary: pd.DataFrame,
    hpo_convergence_summary: pd.DataFrame,
    hpo_top_trials: pd.DataFrame,
    hpo_model_comparison: pd.DataFrame,
    hpo_statistical_analysis: pd.DataFrame,
    hpo_parameter_group_tests: pd.DataFrame,
    hpo_cross_model_tests: pd.DataFrame,
    plots_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    plot_hota_distributions(cleaned_df, plots_dir)
    plot_best_hota(descriptive_df, plots_dir)
    plot_improvements(improvement_df, plots_dir)
    plot_validation_vs_final(descriptive_df, plots_dir)
    plot_deta_vs_assa(decomposition_summary, plots_dir)
    plot_decomposition_improvements(decomposition_improvement_summary, plots_dir)
    plot_decomposition_metrics_across_stages(decomposition_summary, plots_dir)
    plot_best_validation_to_final_decomposition(
        best_validation_to_final_decomposition, plots_dir
    )
    plot_precision_recall_profiles(
        decomposition_summary,
        "best_DetPr",
        "best_DetRe",
        "Best-Run Detection Precision-Recall Profiles",
        "detection_precision_recall_profiles.png",
        plots_dir,
    )
    plot_precision_recall_profiles(
        decomposition_summary,
        "best_AssPr",
        "best_AssRe",
        "Best-Run Association Precision-Recall Profiles",
        "association_precision_recall_profiles.png",
        plots_dir,
    )
    plot_hpo_trial_progression(cleaned_df, plots_dir)
    plot_hpo_ecdf(cleaned_df, plots_dir)
    plot_hpo_stage_summary_overview(hpo_stage_summary, plots_dir)
    plot_hpo_convergence_summary_overview(hpo_convergence_summary, plots_dir)
    plot_hpo_model_comparison_overview(hpo_model_comparison, plots_dir)
    plot_hpo_statistical_analysis_overview(hpo_statistical_analysis, plots_dir)
    plot_hpo_parameter_group_tests_overview(hpo_parameter_group_tests, plots_dir)
    plot_hpo_cross_model_tests_overview(hpo_cross_model_tests, plots_dir)
    plot_hyperparameter_sensitivity(cleaned_df, raw_df, sensitivity_df, plots_dir)


def extract_top_sensitivity_findings(
    sensitivity_df: pd.DataFrame, model: str, top_n: int = 2
) -> list[str]:
    model_sens = sensitivity_df.loc[sensitivity_df["model"] == model].head(top_n)
    findings: list[str] = []
    for _, row in model_sens.iterrows():
        label = row.get("display_parameter") or row.get("parameter")
        if row["parameter_type"] == "numeric":
            range_note = (
                f" Suggested range: {row['suggested_best_range']}."
                if pd.notna(row.get("suggested_best_range"))
                else ""
            )
            findings.append(f"{label}: {row['interpretation']}{range_note}")
        else:
            findings.append(
                f"{label}: {row['interpretation']} Best observed category: {row['best_category']}."
            )
    return findings


def assess_hpo_learning_behavior(
    model_desc: pd.DataFrame, model_var: pd.DataFrame
) -> str:
    if "hyperparameter_tuning" not in model_desc.index:
        return "No hyperparameter tuning stage was available for interpretation."
    if "baseline" not in model_desc.index:
        return "Hyperparameter tuning is present, but baseline is missing so adaptability cannot be benchmarked cleanly."
    tuned_gain = (
        model_desc.loc["hyperparameter_tuning", "max_HOTA"]
        - model_desc.loc["baseline", "max_HOTA"]
    )
    variability_row = model_var.loc["hyperparameter_tuning"] if "hyperparameter_tuning" in model_var.index else None
    cv = (
        variability_row["coefficient_of_variation"]
        if variability_row is not None
        else None
    )
    if tuned_gain > 0 and cv is not None and cv < 0.05:
        return "Tuning improved the best validation result without introducing large variability, which suggests a reasonably navigable search space."
    if tuned_gain > 0 and cv is not None and cv >= 0.05:
        return "Tuning improved the best validation result, but the accompanying variability suggests a sensitive or unstable optimization landscape."
    if tuned_gain <= 0 and cv is not None and cv < 0.05:
        return "Tuning did not materially improve performance and the low variability suggests the model may have saturated early."
    return "Tuning evidence is mixed because gains are modest or variability is too high for a strong conclusion."


def build_model_conclusion_sections(
    descriptive_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    for model in sorted(descriptive_df["model"].dropna().unique()):
        model_desc = descriptive_df.loc[descriptive_df["model"] == model].set_index("stage")
        model_var = variability_df.loc[variability_df["model"] == model].set_index("stage")
        sensitivity_findings = extract_top_sensitivity_findings(sensitivity_df, model)

        baseline_text = "Baseline stage is missing."
        if "baseline" in model_desc.index:
            baseline_text = (
                f"Baseline best HOTA was {format_optional(model_desc.loc['baseline', 'max_HOTA'])} "
                f"from run {model_desc.loc['baseline', 'best_run_name']}."
            )

        finetuning_text = "Finetuning stage is not available for this model."
        if "finetuning" in model_desc.index and "baseline" in model_desc.index:
            delta = model_desc.loc["finetuning", "max_HOTA"] - model_desc.loc["baseline", "max_HOTA"]
            if delta > 0:
                finetuning_text = f"Finetuning improved best HOTA by {delta:.4f} over baseline."
            elif delta < 0:
                finetuning_text = f"Finetuning reduced best HOTA by {abs(delta):.4f} relative to baseline."
            else:
                finetuning_text = "Finetuning produced no observable best-HOTA change relative to baseline."

        hpo_text = assess_hpo_learning_behavior(model_desc, model_var)
        if "hyperparameter_tuning" in model_desc.index and "baseline" in model_desc.index:
            gain = model_desc.loc["hyperparameter_tuning", "max_HOTA"] - model_desc.loc["baseline", "max_HOTA"]
            hpo_text = f"{hpo_text} Best tuned validation gain over baseline was {gain:.4f}."

        stability_text = "Stability could not be assessed because repeated runs were insufficient."
        if "hyperparameter_tuning" in model_var.index:
            row = model_var.loc["hyperparameter_tuning"]
            stability_text = (
                f"HPO stability: {row['robustness_comment']} "
                f"(std={format_optional(row['std_HOTA'])}, cv={format_optional(row['coefficient_of_variation'])})."
            )

        sensitivity_text = (
            " ".join(sensitivity_findings)
            if sensitivity_findings
            else "No interpretable hyperparameter sensitivity signal was available."
        )

        validation_shift_text = "Validation-to-test shift could not be assessed."
        if "final_evaluation" in model_desc.index and "hyperparameter_tuning" in model_desc.index:
            delta = (
                model_desc.loc["final_evaluation", "max_HOTA"]
                - model_desc.loc["hyperparameter_tuning", "max_HOTA"]
            )
            if delta > 0:
                validation_shift_text = f"Final test HOTA exceeded the best validation HOTA by {delta:.4f}."
            elif delta < 0:
                validation_shift_text = f"Final test HOTA fell below the best validation HOTA by {abs(delta):.4f}."
            else:
                validation_shift_text = "Final test HOTA matched the best validation HOTA."

        caveat_parts = []
        for stage in ["baseline", "finetuning", "hyperparameter_tuning", "final_evaluation"]:
            if stage not in model_desc.index:
                caveat_parts.append(f"{stage} is missing")
        if "hyperparameter_tuning" in model_desc.index and model_desc.loc["hyperparameter_tuning", "n_runs"] < 5:
            caveat_parts.append("very few HPO trials were available")
        caveats_text = (
            "Caveats: " + "; ".join(caveat_parts) + "."
            if caveat_parts
            else "Caveats: no additional model-specific caveats beyond the global limitations."
        )

        sections[model] = {
            "baseline": baseline_text,
            "finetuning": finetuning_text,
            "hpo": hpo_text,
            "stability": stability_text,
            "sensitivity": sensitivity_text,
            "validation_shift": validation_shift_text,
            "caveats": caveats_text,
        }
    return sections


def build_recommendations(cross_model_df: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    if cross_model_df.empty:
        return recommendations
    best_final = cross_model_df.loc[cross_model_df["is_best_final_model"]]
    if not best_final.empty:
        recommendations.append(
            f"Prioritize {best_final.iloc[0]['model']} for deployment-oriented follow-up because it achieved the best final test HOTA in this export."
        )
    stable = cross_model_df.loc[cross_model_df["is_most_stable_model"]]
    if not stable.empty:
        recommendations.append(
            f"{stable.iloc[0]['model']} showed the lowest average coefficient of variation, making it a strong candidate when repeatability matters."
        )
    numeric_sensitive = (
        cross_model_df.loc[cross_model_df["is_most_numeric_sensitive_model"]]
        if "is_most_numeric_sensitive_model" in cross_model_df.columns
        else pd.DataFrame()
    )
    categorical_sensitive = (
        cross_model_df.loc[cross_model_df["is_most_categorical_sensitive_model"]]
        if "is_most_categorical_sensitive_model" in cross_model_df.columns
        else pd.DataFrame()
    )
    if not numeric_sensitive.empty:
        recommendations.append(
            f"{numeric_sensitive.iloc[0]['model']} showed the strongest numeric hyperparameter dependence, so budget extra search or stronger priors for continuous tuning ranges."
        )
    if not categorical_sensitive.empty:
        recommendations.append(
            f"{categorical_sensitive.iloc[0]['model']} showed the largest categorical performance spread, so discrete setting choices should be examined carefully in follow-up tuning."
        )
    return recommendations


def build_limitations(
    raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, warnings: list[str]
) -> list[str]:
    limitations = [
        "Only active experiments and active runs are included; deleted entities are intentionally excluded.",
        "Primary analysis defaults to FINISHED runs, so interrupted or RUNNING trials are preserved only in the raw export.",
        "Stage mapping and HOTA extraction rely on explicit logging conventions captured in the CONFIG section.",
        "BoostTrack++ and SiamMOT parent summary runs are preserved in exports but excluded from core comparisons to avoid double-counting.",
        "Hyperparameter tuning trials are not independent random-seed repeats; their distributions reflect both model sensitivity and the chosen search space.",
        "Cross-model HPO comparisons depend on the quality and fairness of each model's search space and tuning budget.",
    ]
    if cleaned_df["seed"].isna().all():
        limitations.append(
            "Seed information was absent in the analyzed runs, so repeated-run interpretation may be limited."
        )
    if warnings:
        limitations.append(
            "Warnings were raised during experiment discovery or run filtering; review the raw export and logs for details."
        )
    if raw_df.empty:
        limitations.append(
            "No runs were retrieved, so all downstream outputs are empty scaffolds."
        )
    return limitations


def build_cross_model_claims(cross_model_df: pd.DataFrame) -> list[str]:
    claims: list[str] = []
    if cross_model_df.empty:
        return claims
    best_final = cross_model_df.loc[cross_model_df["is_best_final_model"]]
    if not best_final.empty:
        row = best_final.iloc[0]
        claims.append(
            f"Best overall model on final test HOTA: {row['model']} with {format_optional(row['final_evaluation_best_HOTA'])}."
        )
    most_improved = cross_model_df.loc[cross_model_df["is_most_improved_model"]]
    if not most_improved.empty:
        row = most_improved.iloc[0]
        claims.append(
            f"Most efficient learner by baseline-to-tuned-validation gain: {row['model']} with a gain of {format_optional(row['baseline_to_tuned_validation_gain'])}."
        )
    most_stable = cross_model_df.loc[cross_model_df["is_most_stable_model"]]
    if not most_stable.empty:
        row = most_stable.iloc[0]
        claims.append(
            f"Most stable model: {row['model']} with average coefficient of variation {format_optional(row['stability_score_cv_mean'])}."
        )
    return claims


def build_hpo_findings(
    hpo_stage_summary: pd.DataFrame,
    hpo_convergence_summary: pd.DataFrame,
    hpo_model_comparison: pd.DataFrame,
    hpo_statistical_analysis: pd.DataFrame,
    hpo_cross_model_tests: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    if hpo_stage_summary.empty:
        return findings
    merged = hpo_stage_summary.merge(hpo_convergence_summary, on="model", how="left")
    for _, row in merged.iterrows():
        findings.append(
            f"{row['model']}: median HOTA {format_optional(row['median_HOTA'])}, "
            f"IQR {format_optional(row['iqr_HOTA'])}, "
            f"{row['proportion_above_baseline']:.1%} of trials above baseline, "
            f"{row['proportion_within_5pct_of_best']:.1%} of trials within 5% of the best result, "
            f"and 95% of the best result was first reached by trial {int(row['first_trial_reaching_95pct_of_best'])}."
        )
    if not hpo_model_comparison.empty:
        best = hpo_model_comparison.loc[hpo_model_comparison["is_best_hpo_model"]]
        if not best.empty:
            findings.append(
                f"Best HPO peak performance came from {best.iloc[0]['model']} with HOTA {format_optional(best.iloc[0]['best_HOTA'])}."
            )
        robust = hpo_model_comparison.loc[hpo_model_comparison["is_most_robust_hpo_model"]]
        if not robust.empty:
            findings.append(
                f"Most robust HPO distribution came from {robust.iloc[0]['model']} with IQR {format_optional(robust.iloc[0]['iqr_HOTA'])}."
            )
        efficient = hpo_model_comparison.loc[hpo_model_comparison["is_most_hpo_efficient_model"]]
        if not efficient.empty:
            findings.append(
                f"Fastest convergence to 95% of best HOTA came from {efficient.iloc[0]['model']} by trial {int(efficient.iloc[0]['first_trial_reaching_95pct_of_best'])}."
            )
    if not hpo_statistical_analysis.empty:
        baseline_rows = hpo_statistical_analysis.loc[
            hpo_statistical_analysis["analysis_name"] == "hpo_vs_baseline"
        ]
        for _, row in baseline_rows.iterrows():
            findings.append(
                f"{row['model']}: the HPO median-minus-baseline difference was {format_optional(row['median_difference_vs_reference'])} "
                f"(95% bootstrap CI {format_optional(row['median_difference_ci95_low'])} to {format_optional(row['median_difference_ci95_high'])}), "
                f"with {row['proportion_above_reference']:.1%} of trials above baseline."
            )
        top_rows = hpo_statistical_analysis.loc[
            hpo_statistical_analysis["analysis_name"] == "top_quartile_vs_rest"
        ]
        for _, row in top_rows.iterrows():
            findings.append(
                f"{row['model']}: the top-quartile HPO region outperformed the remaining trials by a mean HOTA gap of "
                f"{format_optional(row['mean_difference_vs_reference'])} with Mann-Whitney p={format_optional(row['p_value'])}."
            )
    if not hpo_cross_model_tests.empty:
        pairwise = hpo_cross_model_tests.loc[
            hpo_cross_model_tests["comparison_type"] == "pairwise"
        ].sort_values("p_value", na_position="last")
        if not pairwise.empty:
            row = pairwise.iloc[0]
            findings.append(
                f"Across full HPO distributions, {row['model_b']} vs {row['model_a']} showed a median HOTA difference of "
                f"{format_optional(row['median_difference_b_minus_a'])} "
                f"(95% bootstrap CI {format_optional(row['bootstrap_median_diff_ci95_low'])} to {format_optional(row['bootstrap_median_diff_ci95_high'])}; "
                f"Mann-Whitney p={format_optional(row['p_value'])})."
            )
    return findings


def build_decomposition_findings(
    decomposition_summary: pd.DataFrame,
    decomposition_improvement_summary: pd.DataFrame,
    performance_profiles: pd.DataFrame,
    decomposition_cross_model_comparison: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    if not performance_profiles.empty:
        for _, row in performance_profiles.iterrows():
            findings.append(
                f"{row['model']}: at the {row['reference_stage']} stage, the best-run model profile was {row['performance_profile']}, "
                f"with detection {row['detection_profile']} and association {row['association_profile']} precision-recall balance."
            )
    if not decomposition_improvement_summary.empty:
        hpo_rows = decomposition_improvement_summary.loc[
            decomposition_improvement_summary["to_stage"] == "hyperparameter_tuning"
        ]
        for _, row in hpo_rows.iterrows():
            if row.get("primary_driver") is None:
                continue
            findings.append(
                f"{row['model']}: the best-run move to hyperparameter tuning was driven mainly by {row['primary_driver'].replace('_', ' ')}, "
                f"with DetA change {format_optional(row.get('delta_DetA'))} and AssA change {format_optional(row.get('delta_AssA'))}."
            )
    if not decomposition_cross_model_comparison.empty:
        final_stage = decomposition_cross_model_comparison.loc[
            decomposition_cross_model_comparison["stage"] == "final_evaluation"
        ]
        if not final_stage.empty:
            best_det = final_stage.loc[final_stage["is_best_detection_model"]]
            best_ass = final_stage.loc[final_stage["is_best_association_model"]]
            if not best_det.empty:
                findings.append(
                    f"At final evaluation, the strongest detection component came from {best_det.iloc[0]['model']} with DetA {format_optional(best_det.iloc[0]['best_DetA'])}."
                )
            if not best_ass.empty:
                findings.append(
                    f"At final evaluation, the strongest association component came from {best_ass.iloc[0]['model']} with AssA {format_optional(best_ass.iloc[0]['best_AssA'])}."
                )
    return findings


def split_artifact_list(raw_value: Any) -> list[str]:
    if raw_value is None or pd.isna(raw_value):
        return []
    return [item.strip() for item in str(raw_value).split(";") if item.strip()]


def build_question_sections(question_answers: pd.DataFrame) -> tuple[list[str], list[str]]:
    question_sections_markdown: list[str] = []
    question_sections_plain: list[str] = []
    if question_answers.empty:
        return question_sections_markdown, question_sections_plain

    primary_question_rows = question_answers.loc[
        question_answers["question_id"].astype(str).str.fullmatch(r"Q\d+")
    ].copy()
    if primary_question_rows.empty:
        return question_sections_markdown, question_sections_plain

    primary_question_rows["question_number"] = primary_question_rows[
        "question_id"
    ].astype(str).str.removeprefix("Q").astype(int)
    primary_question_rows = primary_question_rows.sort_values(
        ["question_number", "scope"], kind="stable"
    )

    for question_id, group in primary_question_rows.groupby("question_id", sort=False):
        question_text = group.iloc[0]["question"]
        markdown_lines = [f"### {question_id}. {question_text}"]
        plain_lines = [f"{question_id}. {question_text}"]

        for _, qa_row in group.iterrows():
            scope = qa_row.get("scope")
            answer = qa_row.get("answer")
            scope_prefix = f"{scope}: " if pd.notna(scope) and str(scope).strip() else ""
            markdown_lines.append(f"- {scope_prefix}{answer}")
            plain_lines.append(f"- {scope_prefix}{answer}")

        tables = sorted(
            {
                item
                for raw_value in group["supporting_tables"].tolist()
                for item in split_artifact_list(raw_value)
            }
        )
        plots = sorted(
            {
                item
                for raw_value in group["supporting_plots"].tolist()
                for item in split_artifact_list(raw_value)
            }
        )
        if tables:
            markdown_lines.append(
                f"Supported by tables: {', '.join(f'`{table}`' for table in tables)}"
            )
            plain_lines.append(f"Supported by tables: {', '.join(tables)}")
        if plots:
            markdown_lines.append(
                f"Supported by plots: {', '.join(f'`{plot}`' for plot in plots)}"
            )
            plain_lines.append(f"Supported by plots: {', '.join(plots)}")

        question_sections_markdown.append("\n".join(markdown_lines))
        question_sections_plain.append("\n".join(plain_lines))

    return question_sections_markdown, question_sections_plain


def build_technical_summary(
    artifacts: PipelineArtifacts,
    config: dict[str, Any],
) -> tuple[str, str]:
    raw_df = artifacts.raw_runs
    cleaned_df = artifacts.cleaned_runs
    descriptive_df = artifacts.descriptive_summary
    hota_question_answers = artifacts.hota_question_answers
    hyperparameter_question_answers = artifacts.hyperparameter_question_answers
    decomposition_summary = artifacts.decomposition_summary
    decomposition_improvement_summary = artifacts.decomposition_improvement_summary
    best_validation_to_final_decomposition = artifacts.best_validation_to_final_decomposition
    performance_profiles = artifacts.performance_profiles
    decomposition_cross_model_comparison = artifacts.decomposition_cross_model_comparison
    decomposition_question_answers = artifacts.decomposition_question_answers
    cross_model_df = artifacts.cross_model_comparison
    variability_df = artifacts.variability_summary
    sensitivity_df = artifacts.hyperparameter_sensitivity
    hpo_stage_summary = artifacts.hpo_stage_summary
    hpo_convergence_summary = artifacts.hpo_convergence_summary
    hpo_model_comparison = artifacts.hpo_model_comparison
    hpo_statistical_analysis = artifacts.hpo_statistical_analysis
    hpo_cross_model_tests = artifacts.hpo_cross_model_tests

    objective = (
        "Analyze active MLflow runs for MOTIP, BoostTrack++, and SiamMOT using normalized HOTA as the primary metric, "
        "with DetA, AssA, DetPr, DetRe, AssPr, and AssRe as decomposition diagnostics across baseline, finetuning, hyperparameter tuning, and final evaluation stages."
    )
    data_used = [
        f"Active experiment targets: {', '.join(config['experiment_names'])}.",
        f"Retrieved active runs: {len(raw_df)}.",
        f"Primary-analysis runs after filtering: {len(cleaned_df)}.",
    ]
    methodology = [
        "Filtered to active experiments and active runs only, excluding deleted entities by design.",
        "Normalized model and stage labels with model-specific tag and run-name rules.",
        "Selected normalized HOTA and decomposition metric fields per model and stage while preserving the raw source field names.",
        "Used the best observed run within each stage as the default basis for decomposition stage comparisons, while retaining mean decomposition values for HPO landscape context.",
        "Separated raw exported runs from analysis-eligible runs so incomplete and summary-only runs do not inflate comparisons.",
        "Computed descriptive summaries, stage-to-stage improvements, variability statistics, hyperparameter sensitivity, cross-model comparisons, and cautious statistical tests.",
    ]
    model_sections = build_model_conclusion_sections(
        descriptive_df, variability_df, sensitivity_df
    )
    recommendations = build_recommendations(cross_model_df)
    limitations = build_limitations(raw_df, cleaned_df, artifacts.warnings)
    cross_model_lines = build_cross_model_claims(cross_model_df)
    hpo_findings = build_hpo_findings(
        hpo_stage_summary,
        hpo_convergence_summary,
        hpo_model_comparison,
        hpo_statistical_analysis,
        hpo_cross_model_tests,
    )
    decomposition_findings = build_decomposition_findings(
        decomposition_summary,
        decomposition_improvement_summary,
        performance_profiles,
        decomposition_cross_model_comparison,
    )
    hota_question_sections_markdown, hota_question_sections_plain = build_question_sections(
        hota_question_answers
    )
    hyperparameter_question_sections_markdown, hyperparameter_question_sections_plain = build_question_sections(
        hyperparameter_question_answers
    )
    question_sections_markdown, question_sections_plain = build_question_sections(
        decomposition_question_answers
    )

    main_results = [
        f"{model}: {sections['baseline']} {sections['hpo']} {sections['validation_shift']}"
        for model, sections in model_sections.items()
    ]

    markdown_model_sections: list[str] = []
    plain_model_sections: list[str] = []
    for model, sections in model_sections.items():
        markdown_model_sections.extend(
            [
                f"### {model}",
                f"- Baseline performance: {sections['baseline']}",
                f"- Improvement from finetuning: {sections['finetuning']}",
                f"- Improvement from HPO: {sections['hpo']}",
                f"- Stability: {sections['stability']}",
                f"- Sensitivity: {sections['sensitivity']}",
                f"- Validation-to-test shift: {sections['validation_shift']}",
                f"- Caveats: {sections['caveats'].removeprefix('Caveats: ')}",
            ]
        )
        plain_model_sections.extend(
            [
                model,
                f"- Baseline performance: {sections['baseline']}",
                f"- Improvement from finetuning: {sections['finetuning']}",
                f"- Improvement from HPO: {sections['hpo']}",
                f"- Stability: {sections['stability']}",
                f"- Sensitivity: {sections['sensitivity']}",
                f"- Validation-to-test shift: {sections['validation_shift']}",
                f"- Caveats: {sections['caveats'].removeprefix('Caveats: ')}",
                "",
            ]
        )

    markdown_sections = [
        "# Technical Summary",
        "## Objective",
        objective,
        "## Data Used",
        "\n".join(f"- {line}" for line in data_used),
        "## Methodology",
        "\n".join(f"- {line}" for line in methodology),
        "## Main Results",
        "\n".join(
            f"- {line}"
            for line in (
                main_results
                or ["No comparable runs were available for evidence-based findings."]
            )
        ),
        "## Model-by-Model Conclusions",
        "\n".join(markdown_model_sections or ["Insufficient data to compare models by stage."]),
        "## Cross-Model Conclusions",
        "\n".join(
            f"- {line}"
            for line in (
                cross_model_lines
                or ["Cross-model comparison was not possible with the retrieved runs."]
            )
        ),
        "## HPO-Focused Findings",
        "\n".join(
            f"- {line}"
            for line in (
                hpo_findings
                or ["HPO-focused findings were not available because no comparable tuning trials were retrieved."]
            )
        ),
        "## Decomposition Findings",
        "\n".join(
            f"- {line}"
            for line in (
                decomposition_findings
                or ["Decomposition findings were not available because the supporting metrics were missing."]
            )
        ),
        "## Question-Driven HOTA Analysis",
        "\n\n".join(
            hota_question_sections_markdown
            or ["Question-driven HOTA answers were not available."]
        ),
        "## Question-Driven Hyperparameter Analysis",
        "\n\n".join(
            hyperparameter_question_sections_markdown
            or ["Question-driven hyperparameter answers were not available."]
        ),
        "## Question-Driven Decomposition Analysis",
        "\n\n".join(
            question_sections_markdown
            or ["Question-driven decomposition answers were not available."]
        ),
        "## Practical Recommendations",
        "\n".join(
            f"- {line}"
            for line in (
                recommendations
                or ["Collect more finished runs before making deployment decisions."]
            )
        ),
        "## Limitations",
        "\n".join(f"- {line}" for line in limitations),
    ]
    markdown_text = "\n\n".join(markdown_sections) + "\n"

    plain_text_sections = [
        "TECHNICAL SUMMARY",
        "",
        "Objective",
        textwrap.fill(objective, width=100),
        "",
        "Data Used",
        "\n".join(f"- {line}" for line in data_used),
        "",
        "Methodology",
        "\n".join(f"- {line}" for line in methodology),
        "",
        "Main Results",
        "\n".join(
            f"- {line}"
            for line in (
                main_results
                or ["No comparable runs were available for evidence-based findings."]
            )
        ),
        "",
        "Model-by-Model Conclusions",
        "\n".join(plain_model_sections or ["Insufficient data to compare models by stage."]),
        "",
        "Cross-Model Conclusions",
        "\n".join(
            f"- {line}"
            for line in (
                cross_model_lines
                or ["Cross-model comparison was not possible with the retrieved runs."]
            )
        ),
        "",
        "HPO-Focused Findings",
        "\n".join(
            f"- {line}"
            for line in (
                hpo_findings
                or ["HPO-focused findings were not available because no comparable tuning trials were retrieved."]
            )
        ),
        "",
        "Decomposition Findings",
        "\n".join(
            f"- {line}"
            for line in (
                decomposition_findings
                or ["Decomposition findings were not available because the supporting metrics were missing."]
            )
        ),
        "",
        "Question-Driven HOTA Analysis",
        "\n\n".join(
            hota_question_sections_plain
            or ["Question-driven HOTA answers were not available."]
        ),
        "",
        "Question-Driven Hyperparameter Analysis",
        "\n\n".join(
            hyperparameter_question_sections_plain
            or ["Question-driven hyperparameter answers were not available."]
        ),
        "",
        "Question-Driven Decomposition Analysis",
        "\n\n".join(
            question_sections_plain
            or ["Question-driven decomposition answers were not available."]
        ),
        "",
        "Practical Recommendations",
        "\n".join(
            f"- {line}"
            for line in (
                recommendations
                or ["Collect more finished runs before making deployment decisions."]
            )
        ),
        "",
        "Limitations",
        "\n".join(f"- {line}" for line in limitations),
    ]
    plain_text = "\n".join(plain_text_sections) + "\n"
    return markdown_text, plain_text


def format_optional(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.4f}"


def build_export_order(raw_df: pd.DataFrame) -> list[str]:
    ordered = [column for column in CORE_EXPORT_COLUMNS if column in raw_df.columns]
    remaining = [column for column in raw_df.columns if column not in ordered]
    return ordered + sorted(remaining)


def validate_expected_models(
    cleaned_df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> list[str]:
    warnings: list[str] = []
    present_models = set(cleaned_df["model"].dropna().unique())
    for model in config["experiment_names"]:
        if model not in present_models:
            warning = f"Model '{model}' has no analysis-eligible runs after filtering."
            logger.warning(warning)
            warnings.append(warning)
    return warnings


def run_pipeline(args: argparse.Namespace) -> PipelineArtifacts:
    logger = logging.getLogger(__name__)
    config = build_runtime_config(args)
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        logger.info("Using MLflow tracking URI: %s", args.tracking_uri)

    client = MlflowClient()
    experiments, discovery_warnings = fetch_target_experiments(
        client=client,
        experiment_names=config["experiment_names"],
        logger=logger,
    )
    raw_df = build_raw_dataframe(experiments, client, config, logger)
    raw_df = annotate_run_hierarchy(raw_df)
    raw_df = choose_analysis_eligibility(raw_df, config, logger)

    cleaned_df = prepare_cleaned_dataframe(raw_df)
    descriptive_df = build_descriptive_summary(cleaned_df)
    improvement_df = build_improvement_summary(cleaned_df, descriptive_df)
    decomposition_summary = build_decomposition_summary(cleaned_df)
    decomposition_improvement_summary = build_decomposition_improvement_summary(
        decomposition_summary
    )
    best_validation_to_final_decomposition = build_best_validation_to_final_decomposition(
        cleaned_df
    )
    performance_profiles = build_performance_profiles(decomposition_summary)
    decomposition_cross_model_comparison = build_decomposition_cross_model_comparison(
        decomposition_summary
    )
    variability_df = build_variability_summary(cleaned_df)
    sensitivity_df = build_hyperparameter_sensitivity(cleaned_df, raw_df, config)
    hpo_stage_summary = build_hpo_stage_summary(cleaned_df)
    hpo_convergence_summary = build_hpo_convergence_summary(cleaned_df)
    hpo_top_trials = build_hpo_top_trials(cleaned_df)
    hpo_model_comparison = build_hpo_model_comparison(
        hpo_stage_summary, hpo_convergence_summary
    )
    hpo_statistical_analysis = build_hpo_statistical_analysis(cleaned_df)
    hpo_parameter_group_tests = build_hpo_parameter_group_tests(
        cleaned_df, raw_df, config
    )
    hpo_cross_model_tests = build_hpo_cross_model_tests(cleaned_df)
    statistical_df = build_statistical_tests(cleaned_df)
    cross_model_df = build_cross_model_comparison(
        descriptive_df, variability_df, sensitivity_df
    )
    hyperparameter_question_answers = build_hyperparameter_question_answers(
        sensitivity_df,
        hpo_parameter_group_tests,
        hpo_stage_summary,
        hpo_top_trials,
        hpo_model_comparison,
        cross_model_df,
    )
    hota_question_answers = build_hota_question_answers(
        descriptive_df,
        improvement_df,
        variability_df,
        hpo_stage_summary,
        hpo_convergence_summary,
        hpo_model_comparison,
        cross_model_df,
        statistical_df,
    )
    decomposition_question_answers = build_decomposition_question_answers(
        cross_model_df,
        decomposition_summary,
        decomposition_improvement_summary,
        best_validation_to_final_decomposition,
        performance_profiles,
        decomposition_cross_model_comparison,
    )

    warnings = discovery_warnings + validate_expected_models(cleaned_df, config, logger)
    return PipelineArtifacts(
        raw_runs=raw_df,
        cleaned_runs=cleaned_df,
        descriptive_summary=descriptive_df,
        improvement_summary=improvement_df,
        hota_question_answers=hota_question_answers,
        hyperparameter_question_answers=hyperparameter_question_answers,
        decomposition_summary=decomposition_summary,
        decomposition_improvement_summary=decomposition_improvement_summary,
        best_validation_to_final_decomposition=best_validation_to_final_decomposition,
        performance_profiles=performance_profiles,
        decomposition_cross_model_comparison=decomposition_cross_model_comparison,
        decomposition_question_answers=decomposition_question_answers,
        variability_summary=variability_df,
        hyperparameter_sensitivity=sensitivity_df,
        hpo_stage_summary=hpo_stage_summary,
        hpo_convergence_summary=hpo_convergence_summary,
        hpo_top_trials=hpo_top_trials,
        hpo_model_comparison=hpo_model_comparison,
        hpo_statistical_analysis=hpo_statistical_analysis,
        hpo_parameter_group_tests=hpo_parameter_group_tests,
        hpo_cross_model_tests=hpo_cross_model_tests,
        cross_model_comparison=cross_model_df,
        statistical_tests=statistical_df,
        warnings=warnings,
    )


def persist_outputs(
    artifacts: PipelineArtifacts,
    output_dir: Path,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    for png_path in plots_dir.glob("*.png"):
        png_path.unlink()
    raw_df = artifacts.raw_runs.copy()
    cleaned_df = artifacts.cleaned_runs.copy()
    if not raw_df.empty:
        raw_df = raw_df.reindex(columns=build_export_order(raw_df))
    write_dataframe(raw_df, output_dir / "raw_runs_export.csv")
    write_dataframe(cleaned_df, output_dir / "cleaned_runs.csv")
    write_dataframe(
        artifacts.descriptive_summary, output_dir / "descriptive_summary.csv"
    )
    write_dataframe(
        artifacts.improvement_summary, output_dir / "improvement_summary.csv"
    )
    write_dataframe(
        artifacts.hota_question_answers, output_dir / "hota_question_answers.csv"
    )
    write_dataframe(
        artifacts.hyperparameter_question_answers,
        output_dir / "hyperparameter_question_answers.csv",
    )
    write_dataframe(
        artifacts.decomposition_summary, output_dir / "decomposition_summary.csv"
    )
    write_dataframe(
        artifacts.decomposition_improvement_summary,
        output_dir / "decomposition_improvement_summary.csv",
    )
    write_dataframe(
        artifacts.best_validation_to_final_decomposition,
        output_dir / "best_validation_to_final_decomposition.csv",
    )
    write_dataframe(
        artifacts.performance_profiles, output_dir / "performance_profiles.csv"
    )
    write_dataframe(
        artifacts.decomposition_cross_model_comparison,
        output_dir / "decomposition_cross_model_comparison.csv",
    )
    write_dataframe(
        artifacts.decomposition_question_answers,
        output_dir / "decomposition_question_answers.csv",
    )
    write_dataframe(
        artifacts.variability_summary, output_dir / "variability_summary.csv"
    )
    write_dataframe(
        artifacts.hyperparameter_sensitivity,
        output_dir / "hyperparameter_sensitivity.csv",
    )
    write_dataframe(artifacts.hpo_stage_summary, output_dir / "hpo_stage_summary.csv")
    write_dataframe(
        artifacts.hpo_convergence_summary,
        output_dir / "hpo_convergence_summary.csv",
    )
    write_dataframe(artifacts.hpo_top_trials, output_dir / "hpo_top_trials.csv")
    write_dataframe(
        artifacts.hpo_model_comparison,
        output_dir / "hpo_model_comparison.csv",
    )
    write_dataframe(
        artifacts.hpo_statistical_analysis,
        output_dir / "hpo_statistical_analysis.csv",
    )
    write_dataframe(
        artifacts.hpo_parameter_group_tests,
        output_dir / "hpo_parameter_group_tests.csv",
    )
    write_dataframe(
        artifacts.hpo_cross_model_tests,
        output_dir / "hpo_cross_model_tests.csv",
    )
    write_dataframe(
        artifacts.cross_model_comparison, output_dir / "cross_model_comparison.csv"
    )
    write_dataframe(artifacts.statistical_tests, output_dir / "statistical_tests.csv")
    export_plot_tables(artifacts, output_dir)

    markdown_summary, plain_summary = build_technical_summary(artifacts, config)
    (output_dir / "technical_summary.md").write_text(markdown_summary, encoding="utf-8")
    (output_dir / "technical_summary.txt").write_text(plain_summary, encoding="utf-8")

    generate_all_plots(
        cleaned_df=cleaned_df,
        raw_df=raw_df,
        descriptive_df=artifacts.descriptive_summary,
        improvement_df=artifacts.improvement_summary,
        decomposition_summary=artifacts.decomposition_summary,
        decomposition_improvement_summary=artifacts.decomposition_improvement_summary,
        best_validation_to_final_decomposition=artifacts.best_validation_to_final_decomposition,
        sensitivity_df=artifacts.hyperparameter_sensitivity,
        hpo_stage_summary=artifacts.hpo_stage_summary,
        hpo_convergence_summary=artifacts.hpo_convergence_summary,
        hpo_top_trials=artifacts.hpo_top_trials,
        hpo_model_comparison=artifacts.hpo_model_comparison,
        hpo_statistical_analysis=artifacts.hpo_statistical_analysis,
        hpo_parameter_group_tests=artifacts.hpo_parameter_group_tests,
        hpo_cross_model_tests=artifacts.hpo_cross_model_tests,
        plots_dir=plots_dir,
    )
    generate_table_style_plots(artifacts, plots_dir)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        load_dependencies()
    except ImportError as exc:
        raise SystemExit(
            "Missing project dependencies. Run `uv sync` first, then retry the analysis command."
        ) from exc
    config = build_runtime_config(args)
    plots_dir = ensure_output_dirs(args.output_dir)
    artifacts = run_pipeline(args)
    persist_outputs(artifacts, args.output_dir, plots_dir, config)
    logging.getLogger(__name__).info(
        "Analysis complete. Outputs written to %s", args.output_dir
    )


if __name__ == "__main__":
    main()
