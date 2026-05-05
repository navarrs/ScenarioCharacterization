#!/bin/zsh

declare -a scenario_types=(gt ho)
declare -a return_criterion=(average critical)

for scenario_type in "${scenario_types[@]}"; do
    for return_criterion in "${return_criterion[@]}"; do
        uv run -m characterization.run_processor characterizer=ad_individual_features return_criterion="$return_criterion" scenario_type="$scenario_type" paths=test
        uv run -m characterization.run_processor characterizer=ad_individual_scores return_criterion="$return_criterion" scenario_type="$scenario_type" paths=test
        uv run -m characterization.run_processor characterizer=ad_interaction_features return_criterion="$return_criterion" scenario_type="$scenario_type" paths=test
        uv run -m characterization.run_processor characterizer=ad_interaction_scores return_criterion="$return_criterion" scenario_type="$scenario_type" paths=test
        uv run -m characterization.run_processor characterizer=ad_safeshift_scores return_criterion="$return_criterion" scenario_type="$scenario_type" paths=test
    done
done
