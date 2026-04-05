#!/usr/bin/env bash
set -euo pipefail

############################
# Usage
############################
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -p <paths_config>     Specifies the configuration containing the data paths to be used
  -d <meta_dir>         Meta directory where analysis JSON files are copied (default: ./meta)
  -u <output_dir>       Output directory for categorical profiling analyses (default: outputs/categorical_profiler)
  -o                    Overwrite existing results
  -c                    Create metadata for the features
  -m <mode>             Execution mode: scratch or resume (default: resume)
  -s <step>             Repeat a specific step by number (see -l for list); ignores progress file
  -l                    List all steps with their numbers and exit
  -n                    Dry run (print commands, do not execute)
  -h                    Show this help message

Examples:
  # Run with all defaults
  $0

  # Create metadata (c) and/or overwrite (o) existing results
  $0 -c
  $0 -o
  $0 -co

  # Dry run to preview commands
  $0 -n

  # Use custom meta and output directories
  $0 -d ./my_meta -u outputs/my_categorical_profiler

  # Resume from the last completed step (default)
  $0 -m resume

  # Start from scratch (clear progress and run all steps)
  $0 -m scratch

EOF
    exit 1
}

############################
# Defaults
############################
DEFAULT_PATHS_CONFIG="labeling_set"
DEFAULT_RUN_MODE="resume"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROGRESS_FILE="$SCRIPT_DIR/categorical_profiler.progress"

DEFAULT_META_DIR="./meta"
DEFAULT_OUTPUT_DIR="outputs/categorical_profiler"
RAW_FEATURES_ANALYSIS_EXPERIMENT_TAG="raw_features_distribution_analysis"
CAT_FEATURES_ANALYSIS_EXPERIMENT_TAG="cat_features_distribution_analysis"
RAW_SCORES_ANALYSIS_EXPERIMENT_TAG="raw_scores_distribution_analysis"
RAW_SCORES_FROM_CAT_FEATURES_ANALYSIS_EXPERIMENT_TAG="raw_scores_from_cat_features_distribution_analysis"
CAT_SCORES_ANALYSIS_EXPERIMENT_TAG="cat_scores_distribution_analysis"

create_metadata=false
overwrite=false
dry_run=false
list_steps=false
run_mode="$DEFAULT_RUN_MODE"
repeat_step=""
meta_dir="$DEFAULT_META_DIR"
output_dir="$DEFAULT_OUTPUT_DIR"
paths_config="$DEFAULT_PATHS_CONFIG"

############################
# Parse arguments
############################

while getopts ":p:d:u:m:s:conlh" opt; do
    case $opt in
        p) paths_config="$OPTARG" ;;
        d) meta_dir="$OPTARG" ;;
        u) output_dir="$OPTARG" ;;
        m) run_mode="$OPTARG" ;;
        s) repeat_step="$OPTARG" ;;
        o) overwrite=true ;;
        c) create_metadata=true ;;
        l) list_steps=true ;;
        n) dry_run=true ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

############################
# Validate arguments
############################
if [ "$run_mode" != "scratch" ] && [ "$run_mode" != "resume" ]; then
    echo "Invalid mode: $run_mode. Use 'scratch' or 'resume'." >&2
    usage
fi

RAW_FEATURE_ANALYSIS_OUTPUT_DIR="$output_dir/$RAW_FEATURES_ANALYSIS_EXPERIMENT_TAG"
RAW_SCORES_FROM_CAT_FEATURES_ANALYSIS_OUTPUT_DIR="$output_dir/$RAW_SCORES_FROM_CAT_FEATURES_ANALYSIS_EXPERIMENT_TAG"

############################
# Run Categorical Profiler
############################

raw_features_cmd=(
    uv run -m characterization.run_processor
    paths="$paths_config"
    characterizer=safeshift_features
    feature_type=continuous
    create_metadata="$create_metadata"
    overwrite="$overwrite"
)

raw_feature_distribution_analysis_cmd=(
    uv run -m characterization.run_feature_analysis
    paths="$paths_config"
    criteria="['critical_continuous']"
    output_dir="$output_dir"
    add_timestamp=false
    exp_tag="$RAW_FEATURES_ANALYSIS_EXPERIMENT_TAG"
)

cp_raw_feature_analysis_to_meta_cmd=(
    cp "$RAW_FEATURE_ANALYSIS_OUTPUT_DIR"/*.json "$meta_dir/"
)

cat_features_cmd=(
    uv run -m characterization.run_processor
    paths="$paths_config"
    characterizer=safeshift_features
    feature_type=categorical
    overwrite="$overwrite"
)

cat_feature_distribution_analysis_cmd=(
    uv run -m characterization.run_feature_analysis
    paths="$paths_config"
    criteria="['critical_categorical']"
    output_dir="$output_dir"
    add_timestamp=false
    exp_tag="$CAT_FEATURES_ANALYSIS_EXPERIMENT_TAG"
)

raw_scores_cmd=(
    uv run -m characterization.run_processor
    paths="$paths_config"
    characterizer=safeshift_scores
    feature_type=continuous
    score_weighting_method="distance_to_ego_agent"
    overwrite="$overwrite"
)

raw_scores_distribution_analysis_cmd=(
    uv run -m characterization.run_score_analysis
    paths="$paths_config"
    criteria="['critical_continuous']"
    output_dir="$output_dir"
    add_timestamp=false
    exp_tag="$RAW_SCORES_ANALYSIS_EXPERIMENT_TAG"
)

raw_scores_from_cat_features_cmd=(
    uv run -m characterization.run_processor
    paths="$paths_config"
    characterizer=safeshift_scores_categorical
    feature_type=categorical
    score_weighting_method="distance_to_ego_agent"
    overwrite="$overwrite"
)

raw_scores_from_cat_features_distribution_analysis_cmd=(
    uv run -m characterization.run_score_analysis
    paths="$paths_config"
    criteria="['critical_categorical']"
    output_dir="$output_dir"
    add_timestamp=false
    exp_tag="$RAW_SCORES_FROM_CAT_FEATURES_ANALYSIS_EXPERIMENT_TAG"
)

cp_raw_scores_from_cat_features_analysis_to_meta_cmd=(
    cp "$RAW_SCORES_FROM_CAT_FEATURES_ANALYSIS_OUTPUT_DIR"/*.json "$meta_dir/"
)

cat_scores_cmd=(
    uv run -m characterization.run_processor
    paths="$paths_config"
    characterizer=safeshift_scores_categorical
    feature_type=categorical
    score_weighting_method="distance_to_ego_agent"
    categorize_scores=true
    overwrite="$overwrite"
)

cat_scores_distribution_analysis_cmd=(
    uv run -m characterization.run_score_analysis
    paths="$paths_config"
    criteria="['critical_categorical']"
    output_dir="$output_dir"
    add_timestamp=false
    exp_tag="$CAT_SCORES_ANALYSIS_EXPERIMENT_TAG"
)

step_labels=(
    "Raw Feature Computation"
    "Raw Feature Distribution Analysis"
    "Copy Raw Feature Analysis Results to Meta Directory"
    "Categorical Feature Computation"
    "Categorical Feature Distribution Analysis"
    "Raw Scores Computation"
    "Raw Scores Distribution Analysis"
    "Raw Scores from Categorical Features Computation"
    "Raw Scores from Categorical Features Distribution Analysis"
    "Copy Raw Scores from Categorical Features Analysis Results to Meta Directory"
    "Categorical Scores Computation"
    "Categorical Scores Distribution Analysis"
)

step_commands=(
    "raw_features_cmd"
    "raw_feature_distribution_analysis_cmd"
    "cp_raw_feature_analysis_to_meta_cmd"
    "cat_features_cmd"
    "cat_feature_distribution_analysis_cmd"
    "raw_scores_cmd"
    "raw_scores_distribution_analysis_cmd"
    "raw_scores_from_cat_features_cmd"
    "raw_scores_from_cat_features_distribution_analysis_cmd"
    "cp_raw_scores_from_cat_features_analysis_to_meta_cmd"
    "cat_scores_cmd"
    "cat_scores_distribution_analysis_cmd"
)

total_steps=${#step_labels[@]}

if $list_steps; then
    echo "Steps:"
    for ((i = 0; i < total_steps; i++)); do
        echo "  $((i + 1)). ${step_labels[$i]}"
    done
    exit 0
fi

# Run a single step and exit, bypassing the progress file entirely.
if [ -n "$repeat_step" ]; then
    if ! [[ "$repeat_step" =~ ^[0-9]+$ ]] || [ "$repeat_step" -lt 1 ] || [ "$repeat_step" -gt "$total_steps" ]; then
        echo "Invalid step: $repeat_step. Must be between 1 and $total_steps." >&2
        usage
    fi
    label="${step_labels[$((repeat_step - 1))]}"
    declare -n cmd_array="${step_commands[$((repeat_step - 1))]}"
    if $dry_run; then
        echo "Dry run: would repeat step $repeat_step/$total_steps [$label]: ${cmd_array[*]}"
    else
        echo "Repeating step $repeat_step/$total_steps: $label"
        if "${cmd_array[@]}"; then
            echo "Completed step $repeat_step/$total_steps"
        else
            echo "Step failed: $label" >&2
            exit 1
        fi
    fi
    exit 0
fi

last_completed_step=0
if [ "$run_mode" = "scratch" ]; then
    rm -f "$PROGRESS_FILE"
else
    if [ -f "$PROGRESS_FILE" ]; then
        saved_step=$(cat "$PROGRESS_FILE")
        if [[ "$saved_step" =~ ^[0-9]+$ ]]; then
            last_completed_step="$saved_step"
        fi
    fi
fi

if [ "$last_completed_step" -ge "$total_steps" ]; then
    echo "All steps already completed according to progress file: $PROGRESS_FILE"
    exit 0
fi

for ((step_index = last_completed_step + 1; step_index <= total_steps; step_index++)); do
    label="${step_labels[$((step_index - 1))]}"
    declare -n cmd_array="${step_commands[$((step_index - 1))]}"
    if $dry_run; then
        echo "[$step_index/$total_steps] Would run [$label]: ${cmd_array[*]}"
    else
        echo "Running step $step_index/$total_steps: $label"
        if "${cmd_array[@]}"; then
            echo "$step_index" > "$PROGRESS_FILE"
            echo "Completed step $step_index/$total_steps"
        else
            echo "Step failed: $label" >&2
            echo "Resume by running again with -m resume" >&2
            exit 1
        fi
    fi
done
