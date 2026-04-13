You are an expert ML experimentation analyst and Python engineer.

I have an MLflow tracking server running on a VM. It contains experiments for 3 object tracking models:
- BoostTrack++
- SiamMOT
- MOTIP

Each model has its own MLflow experiment, with the exact experiment names:
- MOTIP
- BoostTrack++
- SiamMOT

I want you to generate a complete, reproducible Python analysis pipeline that:
1. connects to MLflow,
2. exports relevant run data,
3. structures and cleans it,
4. performs statistical and comparative analysis,
5. generates plots,
6. exports summary tables to CSV,
7. writes a short technical summary of findings and limitations.

The primary evaluation metric is HOTA. Higher is better.

Use Python scripts, not notebooks. The result must be runnable locally from the command line.

==================================================
PROJECT GOAL
==================================================

Build an end-to-end workflow for analyzing MLflow results for 3 object tracking models across:
- baseline establishment
- finetuning
- hyperparameter tuning
- final evaluation of best tuned model on the test split

The workflow must perform:
1) Structure Your Data
2) Start With Descriptive Analysis
3) Compare Improvements
4) Analyze Variability
5) Hyperparameter Sensitivity Analysis
6) Cross-Model Comparison
7) Interpret Learning Behavior
8) Write Conclusions
9) Statistical Testing
10) Generate plots and export all tables

==================================================
IMPORTANT DOMAIN DETAILS
==================================================

The three models are:
- BoostTrack++
- SiamMOT
- MOTIP

The three experiments in MLflow have these exact names:
- MOTIP
- BoostTrack++
- SiamMOT

Normalize model names to exactly:
- BoostTrack++
- SiamMOT
- MOTIP

The analysis must only include these experiments and only runs that are NOT deleted.

Important requirement about deletion handling:
- I have deleted experiments and runs through the MLflow UI.
- The generated script must exclude deleted experiments and deleted runs.
- Only include experiments from the exact names listed above that are in active lifecycle state.
- Only include runs that are active / not deleted.
- Do not include deleted experiments, deleted parent runs, or deleted child runs.
- If an expected experiment exists but is deleted, warn and skip it.
- If an expected run pattern exists only in deleted runs, do not include it.

Normalize stage names to exactly:
- baseline
- finetuning
- hyperparameter_tuning
- final_evaluation

Because the three models use different stage tag conventions, implement explicit per-model stage normalization logic.

==================================================
EXPERIMENT ORGANIZATION
==================================================

Each model has its own experiment:

1. Experiment name: MOTIP
2. Experiment name: BoostTrack++
3. Experiment name: SiamMOT

The script should search only these three experiments by exact name unless overridden intentionally in config.

==================================================
RUN-NAMING AND STAGE CONVENTIONS
==================================================

Use explicit tag-based mapping first, then run-name-based fallbacks if needed.

------------------------------
MOTIP
------------------------------

Baseline establishment:
- Run name: motip_baseline_evaluation

Finetuning:
- Run name: motip_finetuning

Hyperparameter tuning:
- Run names: motip_hyperparameter_tuning_trial_XXXX
- XXXX corresponds to the trial number
- Example: motip_hyperparameter_tuning_trial_0041 for trial 41
- Each trial is a separate run
- Trial runs that have the status running due to interruption of the machine
  learning pipeline: hspot_hota_optuna_trial_0012, hspot_hota_optuna_trial_0017,
  hspot_hota_optuna_trial_0029, hspot_hota_optuna_trial_0037.

Final evaluation of best hyperparameter run on test split:
- Run name: motip_hyperparameter_tuning_final_evaluation

MOTIP baseline establishment tags:
- stage: baseline_establishment
- dataset_key: HSPOT
- eval_split: val

MOTIP baseline establishment metrics:
- HOTA
- DetA
- AssA
- DetPr
- DetRe
- AssPr
- AssRe
- MOTA
- IDF1

MOTIP finetuning tags:
- stage: finetuning
- dataset_key: HSPOT
- train_split: train
- inference_split: val

MOTIP finetuning metrics:
- loss
- detr_loss
- id_loss
- loss_ce
- class_error
- loss_bbox
- loss_giou
- cardinality_error
- loss_ce_0
- class_error_0
- loss_bbox_0
- loss_giou_0
- cardinality_error_0
- loss_ce_1
- class_error_1
- loss_bbox_1
- loss_giou_1
- cardinality_error_1
- loss_ce_2
- class_error_2
- loss_bbox_2
- loss_giou_2
- cardinality_error_2
- loss_ce_3
- class_error_3
- loss_bbox_3
- loss_giou_3
- cardinality_error_3
- loss_ce_4
- class_error_4
- loss_bbox_4
- loss_giou_4
- cardinality_error_4
- detr_grad_norm
- other_grad_norm
- lr
- max_cuda_mem_MB
- epoch_loss
- epoch_detr_loss
- epoch_id_loss
- epoch_loss_ce
- epoch_class_error
- epoch_loss_bbox
- epoch_loss_giou
- epoch_cardinality_error
- epoch_loss_ce_0
- epoch_class_error_0
- epoch_loss_bbox_0
- epoch_loss_giou_0
- epoch_cardinality_error_0
- epoch_loss_ce_1
- epoch_class_error_1
- epoch_loss_bbox_1
- epoch_loss_giou_1
- epoch_cardinality_error_1
- epoch_loss_ce_2
- epoch_class_error_2
- epoch_loss_bbox_2
- epoch_loss_giou_2
- epoch_cardinality_error_2
- epoch_loss_ce_3
- epoch_class_error_3
- epoch_loss_bbox_3
- epoch_loss_giou_3
- epoch_cardinality_error_3
- epoch_loss_ce_4
- epoch_class_error_4
- epoch_loss_bbox_4
- epoch_loss_giou_4
- epoch_cardinality_error_4
- epoch_detr_grad_norm
- epoch_other_grad_norm
- epoch_lr
- epoch_max_cuda_mem_MB
- epoch
- epoch_HOTA
- epoch_DetA
- epoch_AssA
- epoch_DetPr
- epoch_DetRe
- epoch_AssPr
- epoch_AssRe
- epoch_MOTA
- epoch_IDF1

MOTIP hyperparameter tuning tags:
- stage: hyperparameter_tuning
- dataset_key: HSPOT
- train_split: train
- inference_split: val
- hpo_study_name: hspot_hota_optuna
- hpo_trial_number
- hpo_stage_iter

MOTIP hyperparameter tuning metrics:
- same metrics as MOTIP finetuning above

MOTIP final evaluation tags:
- stage: final_evaluation_from_tuning
- dataset_key: HSPOT
- eval_split: test
- hpo_study_name: hspot_hota_optuna
- hpo_trial_number:
- hpo_stage_iter:

MOTIP final evaluation metrics:
- HOTA
- DetA
- AssA
- DetPr
- DetRe
- AssPr
- AssRe
- MOTA
- IDF1

MOTIP hyperparameters:
- LR
- WEIGHT_DECAY
- LR_BACKBONE_SCALE
- LR_DICTIONARY_SCALE
- LR_WARMUP_EPOCHS
- MAX_CLIP_NORM
- ID_LOSS_WEIGHT
- ASSIGNMENT_PROTOCOL
- DET_THRESH
- NEWBORN_THRESH
- ID_THRESH
- MISS_TOLERANCE
- AREA_THRESH

------------------------------
BoostTrack++
------------------------------

Baseline establishment:
- Parent run name: boosttrack_baseline_evaluation_experiment
- Child run name: boosttrack_baseline_evaluation
- Treat the child validation runs as the actual evaluable runs when metric values are stored there
- Preserve parent-child relationship in exported data

Hyperparameter tuning:
- Parent run name: boosttrack_hyperparameter_tuning_experiment
- Child run names: boosttrack_hyperparameter_tuning_trial_XXXX
- XXXX corresponds to trial number
- Example: boosttrack_hyperparameter_tuning_trial_0075 for trial 75
- Treat the child trial runs as the actual tuning runs when metrics are stored there
- Preserve parent-child relationship in exported data

Final evaluation of best hyperparameter run on test split:
- Child run under parent run boosttrack_hyperparameter_tuning_experiment
- Run name: boosttrack_hyperparameter_tuning_final_evaluation

BoostTrack++ baseline establishment parent run tags:
- dataset: hspot
- benchmark: hspot
- optimizer: optuna_tpe
- pruner: median
- objective: HOTA
- stage: baseline_eval
- fixed_params: {"det_thresh": 0.5, "dlo_boost_coef": 0.6, "iou_threshold": 0.3, "lambda_iou": 0.5, "lambda_mhd": 0.25, "lambda_shape": 0.25, "max_age": 30, "min_hits": 3, "use_dlo_boost": 1, "use_duo_boost": 1}
- best_trial_number: 0
- best_val_exp_name: hspot_baseline_val_trial_000

BoostTrack++ baseline establishment metrics:
- best_val_hota
- best_val_det_a
- best_val_ass_a
- best_val_det_re
- best_val_det_pr
- best_val_ass_re
- best_val_ass_pr
- best_val_loc_a
- best_val_rhota
- best_val_hota_0
- best_val_loc_a_0
- best_val_hota_loc_a_0
- best_val_dets
- best_val_gt_dets
- best_val_i_ds
- best_val_gt_i_ds

BoostTrack++ has no finetuning stage.

BoostTrack++ hyperparameter tuning parent tags
- dataset: hspot
- benchmark: hspot
- optimizer: optuna_tpe
- pruner: median
- objective: HOTA
- best_trial_number:
- best_val_exp_name:

BoostTrack++ hyperparameter tuning parent metrics:
- best_val_hota
- best_val_det_a
- best_val_ass_a
- best_val_det_re
- best_val_det_pr
- best_val_ass_re
- best_val_ass_pr
- best_val_loc_a
- best_val_rhota
- best_val_hota_0
- best_val_loc_a_0
- best_val_hota_loc_a_0
- best_val_dets
- best_val_gt_dets
- best_val_i_ds
- best_val_gt_i_ds
- final_test_hota
- final_test_det_a
- final_test_ass_a
- final_test_det_re
- final_test_det_pr
- final_test_ass_re
- final_test_ass_pr
- final_test_loc_a
- final_test_rhota
- final_test_hota_0
- final_test_loc_a_0
- final_test_hota_loc_a_0
- final_test_dets
- final_test_gt_dets
- final_test_i_ds
- final_test_gt_i_ds

BoostTrack++ hyperparameter tuning trial tags:
- stage: hpo_eval
- hpo_study_name: hspot_hota_optuna
- hpo_trial_number:
- hpo_stage_iter: 1
- trial_state:
- val_exp_name:

BoostTrack++ hyperparameter tuning trial metrics:
- val_hota
- val_det_a
- val_ass_a
- val_det_re
- val_det_pr
- val_ass_re
- val_ass_pr
- val_loc_a
- val_rhota
- val_hota_0
- val_loc_a_0
- val_hota_loc_a_0
- val_dets
- val_gt_dets
- val_i_ds
- val_gt_i_ds

BoostTrack++ final evaluation tags:
- stage: final_eval_best_hpo

BoostTrack++ final evaluation metrics:
- test_hota
- test_det_a
- test_ass_a
- test_det_re
- test_det_pr
- test_ass_re
- test_ass_pr
- test_loc_a
- test_rhota
- test_hota_0
- test_loc_a_0
- test_hota_loc_a_0
- test_dets
- test_gt_dets
- test_i_ds
- test_gt_i_ds

BoostTrack++ hyperparameters:
- det_thresh
- iou_threshold
- min_hits
- max_age
- lambda_iou
- lambda_mhd
- lambda_shape
- dlo_boost_coef
- use_dlo_boost
- use_duo_boost

------------------------------
SiamMOT
------------------------------

Baseline establishment:
- Run name: siammot_baseline_evaluation

Finetuning:
- Training run name: siammot_finetuning

Hyperparameter tuning:
- Parent run name: siammot_hyperparameter_tuning_experiment
- Child run names: siammot_hyperparameter_tuning_trial_XXXX
- XXXX corresponds to the trial number
- Example: - Child run names: siammot_hyperparameter_tuning_trial_XXXX for trial 39
- Treat each child trial run as a separate HPO run
- Preserve parent-child relationship

Final evaluation of best hyperparameter run on test split:
- Separate run name: siammot_hyperparameter_tuning_final_evaluation

SiamMOT baseline establishment tags:
- stage: baseline_eval
- model_name: DLA-34-FPN_box_EMM_MOT_HSPOT
- workflow: baseline_hspot
- dataset_key: MOT_HSPOT
- eval_split: val

SiamMOT baseline establishment metrics:
- infer/mot/num_frames
- infer/mot/mostly_tracked
- infer/mot/partially_tracked
- infer/mot/mostly_lost
- infer/mot/num_switches
- infer/mot/num_false_positives
- infer/mot/num_misses
- infer/mot/mota
- infer/mot/motp
- infer/mot/idf1
- infer/mot/hota/hota
- infer/mot/hota/deta
- infer/mot/hota/assa
- infer/mot/hota/detre
- infer/mot/hota/detpr
- infer/mot/hota/assre
- infer/mot/hota/asspr
- infer/mot/hota/loca
- infer/mot/hota/owta
- infer/mot/hota/hota_0
- infer/mot/hota/loca_0
- infer/mot/hota/hotaloca_0
- infer/mot/hota/dets
- infer/mot/hota/gt_dets
- infer/mot/hota/ids
- infer/mot/hota/gt_ids
- infer/mot/hota
- infer/total_frames
- infer/total_time_sec
- infer/fps
- infer/postprocess/track_score_thresh
- infer/postprocess/min_track_length

SiamMOT finetuning training run tags:
- stage: fine_tune
- model_name: DLA-34-FPN_box_EMM_MOT_HSPOT
- workflow: fine_tune_hspot
- dataset_key: MOT_HSPOT
- train_split: train

SiamMOT finetuning training run metrics:
- train/loss_total
- train/lr
- train/batch_time_sec
- train/data_time_sec
- train/loss_classifier
- train/loss_box_reg
- train/loss_tracker_class
- train/loss_tracker_motion
- train/loss_tracker_center
- train/loss_objectness
- train/loss_rpn_box_reg
- val/epoch
- val/mot/hota/hota
- val/mot/hota/deta
- val/mot/hota/assa
- val/mot/hota/detre
- val/mot/hota/detpr
- val/mot/hota/assre
- val/mot/hota/asspr
- val/mot/hota/loca
- val/mot/hota/owta
- val/mot/hota/hota_0
- val/mot/hota/loca_0
- val/mot/hota/hotaloca_0
- val/mot/hota/dets
- val/mot/hota/gt_dets
- val/mot/hota/ids
- val/mot/hota/gt_ids
- val/mot/hota
- val/total_frames
- val/total_time_sec
- val/fps
- val/postprocess/track_score_thresh
- val/postprocess/min_track_length
- val/objective
- val/best_objective_so_far
- train/total_time_sec
- train/sec_per_iter

SiamMOT hyperparameter tuning parent run tags:
- stage: hpo
- workflow: tune_optuna
- hpo_study_name: hspot_hpo

SiamMOT hyperparameter tuning parent run metrics:
- hpo/best_trial_number
- hpo/best_trial_value
- hpo/completed_trials
- hpo/total_trials

SiamMOT hyperparameter tuning trial run tags:
- stage: hpo_trial
- workflow: tune_optuna:
- hpo_study_name: hspot_hpo:
- hpo_trial_number:
- hpo_parent_run_id:
- latest_stage_name:
- latest_stage_iter:
- hpo_trial_state:
- hpo_final_checkpoint:

SiamMOT's hyperparameter tuning process has 269 steps per epoch. Hence there are
hpo/stage/iter_0000269 and hpo/stage/iter_0000538 metrics.

SiamMOT hyperparameter tuning trial run metrics:
- train/stage/max_iter
- train/stage/effective_num_epochs
- train/stage/steps_per_epoch
- train/loss_total
- train/lr
- train/batch_time_sec
- train/data_time_sec
- train/loss_classifier
- train/loss_box_reg
- train/loss_tracker_class
- train/loss_tracker_motion
- train/loss_tracker_center
- train/loss_objectness
- train/loss_rpn_box_reg
- val/epoch
- val/mot/num_frames
- val/mot/mostly_tracked
- val/mot/partially_tracked
- val/mot/mostly_lost
- val/mot/num_switches
- val/mot/num_false_positives
- val/mot/num_misses
- val/mot/mota
- val/mot/motp
- val/mot/idf1
- val/mot/hota/hota
- val/mot/hota/deta
- val/mot/hota/assa
- val/mot/hota/detre
- val/mot/hota/detpr
- val/mot/hota/assre
- val/mot/hota/asspr
- val/mot/hota/loca
- val/mot/hota/owta
- val/mot/hota/hota_0
- val/mot/hota/loca_0
- val/mot/hota/hotaloca_0
- val/mot/hota/dets
- val/mot/hota/gt_dets
- val/mot/hota/ids
- val/mot/hota/gt_ids
- val/mot/hota
- val/total_frames
- val/total_time_sec
- val/fps
- val/postprocess/track_score_thresh
- val/postprocess/min_track_length
- val/objective
- val/best_objective_so_far
- train/total_time_sec
- train/sec_per_iter
- hpo/objective
- hpo/stage/iter_0000269/infer/mot/hota
- hpo/best_objective_so_far
- hpo/stage/iter_0000269/infer/fps
- hpo/stage/iter_0000269/infer/mot/hota/assa
- hpo/stage/iter_0000269/infer/mot/hota/asspr
- hpo/stage/iter_0000269/infer/mot/hota/assre
- hpo/stage/iter_0000269/infer/mot/hota/deta
- hpo/stage/iter_0000269/infer/mot/hota/detpr
- hpo/stage/iter_0000269/infer/mot/hota/detre
- hpo/stage/iter_0000269/infer/mot/hota/dets
- hpo/stage/iter_0000269/infer/mot/hota/gt_dets
- hpo/stage/iter_0000269/infer/mot/hota/gt_ids
- hpo/stage/iter_0000269/infer/mot/hota/hota
- hpo/stage/iter_0000269/infer/mot/hota/hota_0
- hpo/stage/iter_0000269/infer/mot/hota/hotaloca_0
- hpo/stage/iter_0000269/infer/mot/hota/ids
- hpo/stage/iter_0000269/infer/mot/hota/loca
- hpo/stage/iter_0000269/infer/mot/hota/loca_0
- hpo/stage/iter_0000269/infer/mot/hota/owta
- hpo/stage/iter_0000269/infer/mot/idf1
- hpo/stage/iter_0000269/infer/mot/mostly_lost
- hpo/stage/iter_0000269/infer/mot/mostly_tracked
- hpo/stage/iter_0000269/infer/mot/mota
- hpo/stage/iter_0000269/infer/mot/motp
- hpo/stage/iter_0000269/infer/mot/num_false_positives
- hpo/stage/iter_0000269/infer/mot/num_frames
- hpo/stage/iter_0000269/infer/mot/num_misses
- hpo/stage/iter_0000269/infer/mot/num_switches
- hpo/stage/iter_0000269/infer/mot/partially_tracked
- hpo/stage/iter_0000269/infer/postprocess/min_track_length
- hpo/stage/iter_0000269/infer/postprocess/track_score_thresh
- hpo/stage/iter_0000269/infer/total_frames
- hpo/stage/iter_0000269/infer/total_time_sec
- hpo/stage/iter_0000538/infer/mot/hota
- hpo/stage/iter_0000538/infer/fps
- hpo/stage/iter_0000538/infer/mot/hota/assa
- hpo/stage/iter_0000538/infer/mot/hota/asspr
- hpo/stage/iter_0000538/infer/mot/hota/assre
- hpo/stage/iter_0000538/infer/mot/hota/deta
- hpo/stage/iter_0000538/infer/mot/hota/detpr
- hpo/stage/iter_0000538/infer/mot/hota/detre
- hpo/stage/iter_0000538/infer/mot/hota/dets
- hpo/stage/iter_0000538/infer/mot/hota/gt_dets
- hpo/stage/iter_0000538/infer/mot/hota/gt_ids
- hpo/stage/iter_0000538/infer/mot/hota/hota
- hpo/stage/iter_0000538/infer/mot/hota/hota_0
- hpo/stage/iter_0000538/infer/mot/hota/hotaloca_0
- hpo/stage/iter_0000538/infer/mot/hota/ids
- hpo/stage/iter_0000538/infer/mot/hota/loca
- hpo/stage/iter_0000538/infer/mot/hota/loca_0
- hpo/stage/iter_0000538/infer/mot/hota/owta
- hpo/stage/iter_0000538/infer/mot/idf1
- hpo/stage/iter_0000538/infer/mot/mostly_lost
- hpo/stage/iter_0000538/infer/mot/mostly_tracked
- hpo/stage/iter_0000538/infer/mot/mota
- hpo/stage/iter_0000538/infer/mot/motp
- hpo/stage/iter_0000538/infer/mot/num_false_positives
- hpo/stage/iter_0000538/infer/mot/num_frames
- hpo/stage/iter_0000538/infer/mot/num_misses
- hpo/stage/iter_0000538/infer/mot/num_switches
- hpo/stage/iter_0000538/infer/mot/partially_tracked
- hpo/stage/iter_0000538/infer/postprocess/min_track_length
- hpo/stage/iter_0000538/infer/postprocess/track_score_thresh
- hpo/stage/iter_0000538/infer/total_frames
- hpo/stage/iter_0000538/infer/total_time_sec
- hpo/final_objective
- hpo/best_objective

SiamMOT final evaluation tags:
- stage: final_eval_best_hpo
- workflow: tune_optuna_final_eval
- hpo_study_name: hspot_hpo
- hpo_best_trial_number:
- dataset_key: MOT_HSPOT
- eval_split: test

SiamMOT final evaluation metrics:
- infer/mot/num_frames
- infer/mot/mostly_tracked
- infer/mot/partially_tracked
- infer/mot/mostly_lost
- infer/mot/num_switches
- infer/mot/num_false_positives
- infer/mot/num_misses
- infer/mot/mota
- infer/mot/motp
- infer/mot/idf1
- infer/mot/hota/hota
- infer/mot/hota/deta
- infer/mot/hota/assa
- infer/mot/hota/detre
- infer/mot/hota/detpr
- infer/mot/hota/assre
- infer/mot/hota/asspr
- infer/mot/hota/loca
- infer/mot/hota/owta
- infer/mot/hota/hota_0
- infer/mot/hota/loca_0
- infer/mot/hota/hotaloca_0
- infer/mot/hota/dets
- infer/mot/hota/gt_dets
- infer/mot/hota/ids
- infer/mot/hota/gt_ids
- infer/mot/hota
- infer/total_frames
- infer/total_time_sec
- infer/fps
- infer/postprocess/track_score_thresh
- infer/postprocess/min_track_length

SiamMOT hyperparameters:
- BASE_LR
- WEIGHT_DECAY
- TRACK_THRESH
- START_TRACK_THRESH
- RESUME_TRACK_THRESH
- MAX_DORMANT_FRAMES
- TRACK_SCORE_THRESH
- MIN_TRACK_LENGTH
These hyperparameters are registered in MLflow as:
- optuna.solver_base_lr
- optuna.solver_weight_decay
- optuna.model_track_thresh
- optuna.model_start_track_thresh
- optuna.model_resume_track_thresh
- optuna.model_max_dormant_frames
- optuna.infer_track_score_thresh
- optuna.infer_min_track_length

==================================================
PRIMARY METRIC NORMALIZATION
==================================================

The script must support different HOTA metric field names across models/stages.

Primary normalized metric to use in analysis: HOTA

Metric extraction preferences by model/stage:

MOTIP:
- Prefer HOTA
- Fallback: epoch_HOTA only if final HOTA is not present

BoostTrack++:
- Baseline and HPO validation metric: best_val_hota
- Final test metric: test_hota

SiamMOT:
- Prefer infer/mot/hota/hota
- Fallback: infer/mot/hota if needed
- For HPO runs, also consider hpo/objective and hpo/final_objective only as supplementary fields, not primary, unless configured otherwise
- Final evaluation on test should still use the actual evaluation HOTA metric if present

The script must create:
- a normalized HOTA column
- a column indicating which raw metric field was used

==================================================
STAGE NORMALIZATION RULES
==================================================

Implement explicit stage normalization rules, including per-model handling.

Normalize to:
- baseline
- finetuning
- hyperparameter_tuning
- final_evaluation

Suggested mapping:

MOTIP:
- baseline_establishment -> baseline
- finetuning -> finetuning
- hyperparameter_tuning -> hyperparameter_tuning
- final_evaluation_from_tuning -> final_evaluation

BoostTrack++:
- baseline_eval -> baseline
- hpo_eval -> hyperparameter_tuning
- final_eval_best_hpo -> final_evaluation
- no finetuning stage exists for this model

SiamMOT:
- baseline_eval -> baseline
- fine_tune -> finetuning_training_only
- fine_tune_eval -> finetuning
- hpo_trial -> hyperparameter_tuning
- final_eval_best_hpo -> final_evaluation

Important:
- SiamMOT fine_tune is a training run, not the final evaluation run for that stage
- For SiamMOT finetuning-stage performance comparison, use fine_tune_eval as the evaluation-stage run if available
- Preserve training-only runs in raw export, but use evaluation runs for stage-level metric comparisons where appropriate
- BoostTrack++ has no finetuning stage, so comparisons must handle missing finetuning gracefully

==================================================
PARENT-CHILD RUN HANDLING
==================================================

The script must handle parent and child runs correctly.

Requirements:
- Export parent_run_id if present
- Preserve run hierarchy metadata
- For BoostTrack++ and SiamMOT HPO, detect parent-child relationships
- Use child runs as the main evaluable units when metric values live in the child runs
- Do not double-count parent summary runs as trial runs unless they contain unique evaluable information intended for analysis
- Document the logic clearly in code and README

==================================================
ANALYSIS RULES
==================================================

1. Primary metric for comparison is normalized HOTA.
2. Preserve additional metrics like DetA, AssA, MOTA, IDF1 for optional supplementary analysis.
3. Treat baseline, finetuning, hyperparameter tuning, and final evaluation as distinct stages.
4. If there are multiple runs per stage, analyze both:
   - mean performance
   - best-run performance
5. If seeds are available, treat them as repeated runs.
6. If sample sizes are very small, avoid overclaiming statistical significance.
7. Code must handle missing tags, params, or metrics gracefully.
8. All conclusions must be tied to computed evidence.
9. The script must gracefully handle models with missing stages, especially BoostTrack++ lacking finetuning.
10. Final evaluation on test split should be analyzed separately from validation-stage HPO comparisons, but included in summary outputs.

==================================================
EXPECTED INPUTS
==================================================

Make these configurable through argparse or a config section:
- MLflow tracking URI
- output directory
- exact experiment names, defaulting to:
  - MOTIP
  - BoostTrack++
  - SiamMOT
- metric name preferences
- model identification mapping
- stage identification mapping
- preferred hyperparameters per model
- optional dataset filter
- optional run filters

==================================================
REQUIRED DELIVERABLES
==================================================

Generate code that does all of the following:

A) EXPORT DATA FROM MLFLOW
- Connect to MLflow tracking server.
- Retrieve experiments by exact names:
  - MOTIP
  - BoostTrack++
  - SiamMOT
- Only include experiments whose lifecycle_stage is active.
- Retrieve only runs whose lifecycle stage is active / not deleted.
- Export a raw CSV containing all retrieved runs and selected fields.
- Flatten params, metrics, and tags.
- Include at least:
  - run_id
  - experiment_id
  - experiment_name
  - run_name
  - status
  - lifecycle_stage
  - start_time
  - end_time
  - parent_run_id if present
  - model
  - raw_stage
  - normalized_stage
  - normalized_HOTA
  - HOTA source column used
  - dataset_key
  - dataset
  - benchmark
  - train_split
  - inference_split
  - eval_split
  - seed if present
  - hpo_study_name if present
  - hpo_trial_number if present
  - hpo_stage_iter if present
  - trial_state / hpo_trial_state if present
  - all detected hyperparameters for each model

B) STRUCTURE YOUR DATA
- Build a clean analysis DataFrame with at least:
  - model
  - stage
  - HOTA
  - run_id
  - run_name
  - seed
  - dataset metadata
  - selected hyperparameters
  - whether the run is a parent or child run
- Standardize stage labels into:
  - baseline
  - finetuning
  - hyperparameter_tuning
  - final_evaluation
- Prefer exact tag-based mapping for stage.
- Derive model primarily from experiment name, with fallback to tags/run_name if needed.
- Validate that the expected models are present:
  - BoostTrack++
  - SiamMOT
  - MOTIP
- Warn instead of hard failing if some are missing.
- Handle duplicate runs sensibly.
- Report excluded runs and reasons.
- Save cleaned data to CSV.

C) DESCRIPTIVE ANALYSIS
For each model and stage compute:
- number of runs
- mean HOTA
- median HOTA
- std HOTA
- variance
- min HOTA
- max HOTA
- best run ID
- best run name
- 95% confidence interval where feasible

Export to CSV.

D) COMPARE IMPROVEMENTS
For each model compute both best-run and mean-run comparisons where stages exist:
- finetuning - baseline
- hyperparameter_tuning - finetuning
- hyperparameter_tuning - baseline
- final_evaluation - best hyperparameter_tuning validation result
- final_evaluation - baseline

Include:
- absolute improvement
- relative improvement percentage

Important:
- Handle missing finetuning gracefully for BoostTrack++
- Separate validation-stage comparisons from final test evaluation comparisons

Export to CSV.

E) ANALYZE VARIABILITY
For each model and stage compute:
- std
- variance
- coefficient of variation where meaningful
- range
- IQR where feasible

Use this to characterize:
- robustness
- hyperparameter sensitivity
- whether results are stable or noisy

Export to CSV.

F) HYPERPARAMETER SENSITIVITY ANALYSIS
For hyperparameter_tuning runs:
- analyze each model separately
- use the known hyperparameter lists for each model as priority fields
- also detect additional useful params automatically
- convert numeric params where possible
- compute:
  - Pearson correlation for numeric params where appropriate
  - Spearman correlation for monotonic relationships
  - grouped summaries for categorical params
- rank hyperparameters by apparent association with HOTA
- infer candidate best-performing ranges for top numeric hyperparameters
- create plots:
  - scatter plots for numeric hyperparameters vs HOTA
  - boxplots/barplots for categorical hyperparameters
- keep analysis interpretable and elaborate

Export sensitivity tables to CSV.

G) CROSS-MODEL COMPARISON
Compare all three models on:
- baseline HOTA
- finetuning HOTA where available
- hyperparameter tuning HOTA
- final evaluation HOTA on test split
- total gain from baseline to tuned validation
- total gain from baseline to final test evaluation
- best final HOTA
- mean final HOTA where meaningful
- stability
- sensitivity to hyperparameters

Identify:
- best baseline model
- most improved model
- most stable model
- most sensitive to hyperparameters
- best final model on test split

Export to CSV.

H) INTERPRET LEARNING BEHAVIOR
Generate evidence-based narrative findings that discuss:
- whether finetuning improved performance meaningfully
- whether tuning improved optimization or mostly increased variance
- whether a model saturated early
- whether a model had poor baseline but strong adaptability
- whether high variance suggests an unstable optimization landscape
- whether the model seems robust across hyperparameters
- how final test evaluation compares with validation-stage tuning results
- caveats from limited sample size, missing stages, parent-child logging structure, or inconsistent logging

These conclusions must be based on computed results, not generic wording.

I) WRITE CONCLUSIONS
Produce an elaborate technical report in both Markdown and plain text containing:
1. objective
2. data used
3. methodology
4. main results
5. model-by-model conclusions
6. cross-model conclusions
7. practical recommendations
8. limitations

Write the technical report in a manner that is appropriate for a master's
thesis, but not bombastic.

J) STATISTICAL TESTING
Where sample sizes allow, perform within-model stage comparisons:
- baseline vs finetuning
- finetuning vs hyperparameter_tuning
- baseline vs hyperparameter_tuning
- best tuned validation vs final test evaluation where it makes conceptual sense, with clear caution that val and test serve different purposes

Use sensible methods:
- Welch t-test if assumptions are reasonably acceptable
- Mann–Whitney U if nonparametric comparison is more appropriate
- bootstrap confidence intervals for mean difference where useful

Also report:
- p-values
- effect sizes where feasible
- warning notes for very small sample sizes or incomparable group definitions

Do not overclaim significance if there are very few runs.

Export test results to CSV.

K) PLOTS
Generate readable PNG plots:
1. mean HOTA by model and stage
2. HOTA distributions by model and stage
3. best HOTA by stage for each model
4. improvement plots across stages
5. top hyperparameter sensitivity plots per model
6. optional heatmap of numeric hyperparameter correlations with HOTA
7. separate comparison plot for validation best vs final test evaluation

Display the values that are normally only visible
when hovering over a part of the plot.

Use clear filenames and readable labels.

L) OUTPUT ARTIFACTS
At minimum produce:
- raw_runs_export.csv
- cleaned_runs.csv
- descriptive_summary.csv
- improvement_summary.csv
- variability_summary.csv
- hyperparameter_sensitivity.csv
- cross_model_comparison.csv
- statistical_tests.csv
- technical_summary.md
- technical_summary.txt
- plots/*.png

M) CODE QUALITY
- Use argparse.
- Include a main entry point.
- Add docstrings and comments.
- Include logging.
- Make the script robust and easy to adapt.
- Prefer a single self-contained Python script unless modularization is clearly better.

==================================================
PYTHON VERSIONING AND DEPENDENCY MANAGEMENT
==================================================

Use:
- pyenv for Python version management
- uv for virtual environment creation, dependency management, locking, and running commands

This is a hard requirement.

Please generate a project that is ready to use with pyenv and uv.

Requirements:
1. Choose and document a modern stable Python version compatible with the needed libraries, for example Python 3.11.x unless there is a strong reason to choose another version.
2. Include a `.python-version` file for pyenv.
3. Include a `pyproject.toml` configured for uv-managed dependencies.
4. Include a `uv.lock` only if you choose to provide a realistic lockfile; otherwise explain that it should be generated locally with `uv lock`.
5. Do NOT use requirements.txt as the primary dependency mechanism.
6. You may include an optional requirements.txt export only if clearly marked as secondary/generated, but the main setup must be pyenv + uv.
7. README setup instructions must use pyenv and uv commands.
8. The code should be runnable with `uv run ...`.

Provide setup instructions such as:
- pyenv install 3.11.x
- pyenv local 3.11.x
- uv venv
- source .venv/bin/activate
- uv sync
- uv run python analyze_mlflow_tracking.py ...

If a package is optional, put it in an optional dependency group if appropriate.

==================================================
IMPLEMENTATION PREFERENCES
==================================================

Preferred libraries:
- mlflow
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pathlib
- argparse
- logging
- json
- re
- textwrap

Use statsmodels only if helpful, but keep dependencies reasonable.

Preferred project structure:
- analyze_mlflow_tracking.py
- pyproject.toml
- .python-version
- README.md

Optionally:
- uv.lock

A single self-contained script is preferred unless modularization is clearly better.

==================================================
CONFIG REQUIREMENTS
==================================================

At the top of the script, create a clearly marked CONFIG section that allows me to edit:
- experiment_names
- metric field preferences by model/stage
- fallback metric field names
- stage normalization mapping
- model aliases
- model detection rules
- preferred hyperparameters per model
- tag/param candidates for seed
- dataset filter defaults
- regex rules for parsing run_name if needed
- parent-child handling rules
- whether to include final_evaluation stage in core comparisons

Use these defaults:

Experiment names:
- MOTIP
- BoostTrack++
- SiamMOT

Preferred hyperparameters:
- MOTIP: LR, WEIGHT_DECAY, LR_BACKBONE_SCALE, LR_DICTIONARY_SCALE, LR_WARMUP_EPOCHS, MAX_CLIP_NORM, ID_LOSS_WEIGHT, ASSIGNMENT_PROTOCOL, DET_THRESH, NEWBORN_THRESH, ID_THRESH, MISS_TOLERANCE, AREA_THRESH
- BoostTrack++: det_thresh, iou_threshold, min_hits, max_age, lambda_iou, lambda_mhd, lambda_shape, dlo_boost_coef, use_dlo_boost, use_duo_boost
- SiamMOT: BASE_LR, WEIGHT_DECAY, TRACK_THRESH, START_TRACK_THRESH, RESUME_TRACK_THRESH, MAX_DORMANT_FRAMES, TRACK_SCORE_THRESH, MIN_TRACK_LENGTH

Stage mapping defaults:
- MOTIP:
  - baseline_establishment -> baseline
  - finetuning -> finetuning
  - hyperparameter_tuning -> hyperparameter_tuning
  - final_evaluation_from_tuning -> final_evaluation
- BoostTrack++:
  - baseline_eval -> baseline
  - hpo_eval -> hyperparameter_tuning
  - final_eval_best_hpo -> final_evaluation
- SiamMOT:
  - baseline_eval -> baseline
  - fine_tune -> finetuning_training_only
  - fine_tune_eval -> finetuning
  - hpo_trial -> hyperparameter_tuning
  - final_eval_best_hpo -> final_evaluation

Metric mapping defaults:
- MOTIP:
  - primary: HOTA
  - fallback: epoch_HOTA
- BoostTrack++:
  - baseline/hpo: best_val_hota
  - final_evaluation: test_hota
- SiamMOT:
  - primary: infer/mot/hota/hota
  - fallback: infer/mot/hota

==================================================
SPECIAL HANDLING
==================================================

Special handling for MOTIP:
- For finetuning and hyperparameter tuning, include optional handling for training dynamics if epoch-level metrics are available.
- However, keep the main comparison based on final HOTA.
- If MLflow metric history access is easy to implement, optionally provide a helper for pulling metric histories for epoch_HOTA and epoch_loss-like metrics.
- If metric histories are not available or would overly complicate the implementation, document that clearly and proceed with final logged metrics.

Special handling for BoostTrack++:
- Parse `fixed_params` JSON tag where useful for baseline parameter extraction.
- Correctly distinguish parent summary runs from child evaluable trial runs.
- Use child trial metrics as the main observations when appropriate.

Special handling for SiamMOT:
- Distinguish training-only runs from evaluation runs.
- Use evaluation runs for stage-level performance analysis when possible.
- Keep HPO objective metrics as supplementary unless explicitly configured otherwise.

==================================================
REQUIRED BEHAVIOR FOR AMBIGUITIES
==================================================

Do not ask me follow-up questions.

Instead:
- implement reasonable defaults
- put assumptions in a CONFIG section
- add TODO comments where I may need to adapt mappings
- prefer explicit MLflow tags over inferred values
- derive model from experiment name first
- if a field is missing, continue and log a warning
- if a stage does not exist for a model, handle it gracefully without failing

==================================================
README REQUIREMENTS
==================================================

Include example usage like:

uv run python analyze_mlflow_tracking.py \
  --tracking-uri http://MY_VM:5000 \
  --output-dir outputs

The script should default to the three exact experiment names above, but also allow override through CLI/config.

Also include examples for:
- explicitly passing experiment names
- filtering by dataset_key HSPOT or MOT_HSPOT
- overriding the primary metric
- analyzing only active experiments and active runs

README must include:
1. pyenv setup
2. uv setup
3. project bootstrap
4. dependency installation with uv
5. running the analysis
6. where outputs are written
7. how to adapt the CONFIG section for different MLflow metadata conventions
8. how deleted experiments and deleted runs are excluded

==================================================
FINAL OUTPUT FROM YOU
==================================================

Please generate:
1. full Python code
2. pyproject.toml
3. .python-version
4. README.md
5. helper functions as needed
6. example CLI usage

If you choose to include uv.lock, make it consistent with pyproject.toml. If not, say that it should be generated locally with `uv lock`.

Make the result production-usable, clean, and easy to adapt.