# High-Street Property Occupancy Tracking Experiment Analysis

This project provides a reproducible command-line analysis pipeline for MLflow experiments covering:

- `MOTIP`
- `BoostTrack++`
- `SiamMOT`

It exports active runs, normalizes model/stage conventions, filters out deleted and incomplete runs from core comparisons by default, computes analysis tables, generates plots, and writes technical summaries.

## Project Files

- [analyze_mlflow_tracking.py](/Users/tom/Documents/Master%20of%20Informatics%20-%20Applied%20Data%20Science/Onderzoek/high-street-property-occupancy-tracking-experiment-analysis/analyze_mlflow_tracking.py)
- [pyproject.toml](/Users/tom/Documents/Master%20of%20Informatics%20-%20Applied%20Data%20Science/Onderzoek/high-street-property-occupancy-tracking-experiment-analysis/pyproject.toml)
- [.python-version](/Users/tom/Documents/Master%20of%20Informatics%20-%20Applied%20Data%20Science/Onderzoek/high-street-property-occupancy-tracking-experiment-analysis/.python-version)

## Python Setup

This repo is set up for `pyenv` and `uv`.

```bash
pyenv install 3.11.9
pyenv local 3.11.9
uv venv
source .venv/bin/activate
uv sync
```

`uv.lock` is not committed here. Generate it locally with:

```bash
uv lock
```

## Running The Analysis

Basic usage:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs
```

You can also use the console entry point:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs
```

Override the experiment names explicitly:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs \
  --experiment-names MOTIP "BoostTrack++" SiamMOT
```

Filter by dataset metadata while still preserving filtered-out runs in the raw export with exclusion flags:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs_hspot \
  --dataset-filter HSPOT MOT_HSPOT
```

Override the preferred primary metric field for a specific model/stage:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs_override \
  --metric-override "SiamMOT:hyperparameter_tuning=val/mot/hota/hota,hpo/final_objective"
```

Include unfinished active runs in primary analysis:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs_include_unfinished \
  --include-unfinished-in-primary-analysis
```

Exclude final evaluation runs from the core cleaned comparison dataset:

```bash
uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs_no_final \
  --exclude-final-evaluation-from-core-comparisons
```

## Output Artifacts

The script writes these files into the chosen output directory:

- `raw_runs_export.csv`
- `cleaned_runs.csv`
- `descriptive_summary.csv`
- `improvement_summary.csv`
- `variability_summary.csv`
- `hyperparameter_sensitivity.csv`
- `cross_model_comparison.csv`
- `statistical_tests.csv`
- `technical_summary.md`
- `technical_summary.txt`
- `plots/*.png`

## Deleted And Incomplete Runs

The pipeline is designed to avoid accidental contamination of the analysis set:

- Only experiments in the active MLflow lifecycle state are selected.
- Only active runs are retrieved from those experiments.
- Deleted experiments and deleted runs are skipped.
- If an expected experiment exists only in deleted state, the script logs a warning and skips it.
- Active but unfinished runs stay in `raw_runs_export.csv` and are flagged with `is_incomplete_run`.
- Primary analysis includes only `FINISHED` runs by default.

## Parent-Child Run Handling

The script uses a model-specific run-selection layer:

- Raw exports preserve parent-child metadata such as `parent_run_id`, `is_parent_run`, `is_child_run`, and `run_role`.
- BoostTrack++ parent summary runs are kept in the raw export but excluded from core comparisons when child runs carry the evaluable metrics.
- SiamMOT HPO parent summary runs are preserved but excluded from stage-level HPO comparisons.
- Cleaned analysis outputs use only analysis-eligible runs so study summaries do not inflate trial counts.

## CONFIG Section

The main adaptation point is the clearly marked `CONFIG` section near the top of [analyze_mlflow_tracking.py](/Users/tom/Documents/Master%20of%20Informatics%20-%20Applied%20Data%20Science/Onderzoek/high-street-property-occupancy-tracking-experiment-analysis/analyze_mlflow_tracking.py).

You can edit:

- default experiment names
- model aliases and run-name detection rules
- stage normalization mappings
- metric field preferences and fallbacks
- preferred hyperparameters per model
- SiamMOT parameter-name remapping
- seed tag/param candidates
- dataset filter defaults
- whether to include final evaluation in core comparisons
- whether to restrict primary analysis to finished runs

## Notes

- The primary normalized metric is `HOTA`.
- The script keeps the raw metric source field in `normalized_HOTA_source`.
- Metric histories are not pulled from MLflow; the analysis uses the final metrics available on each run.
- The plotting and statistical outputs are intentionally cautious when sample sizes are small.
