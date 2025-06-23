#!/bin/zsh

uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["gt"]' tag=gt
uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["gt"]' 'criteria=["average"]' tag=gt_average
uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["gt"]' 'criteria=["critical"]' tag=gt_critical

uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["ho"]' tag=ho
uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["ho"]' 'criteria=["average"]' tag=ho_average
uv run -m characterization.viz_scores_pdf paths=test 'scenario_types=["ho"]' 'criteria=["critical"]' tag=ho_critical

uv run -m characterization.viz_scores_pdf paths=test 'scores=["individual"]' tag=individual
uv run -m characterization.viz_scores_pdf paths=test 'scores=["interaction"]' tag=interaction
uv run -m characterization.viz_scores_pdf paths=test 'scores=["safeshift"]' tag=safeshift
