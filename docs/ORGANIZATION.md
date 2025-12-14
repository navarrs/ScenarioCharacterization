# Repository Organization

## Configuration Files (Using Hydra)

The main configuration files are:

1. **`run_processor.yaml`**
   Used for computing scenario features and scores.

2. **`run_analysis.yaml`**
   Used for analyzing features and scores distributions.

3. **`viz_scores_pdf.yaml`**
   Used for processing pre-computed scores, calculating a density function over the scored scenarios, and providing scenario visualizations.


Both configuration files are built hierarchically from the following components:

- **`characterizer`**: Specifies the type of characterization to run (e.g., features, scores).
- **`dataset`**: Defines which dataset adapter to use.
- **`paths`**: Sets the input and output data paths.
- **`processor`**: Determines the type of processor to run. Currently, `feature` and `score` processors are supported.
- **`viz`**: Configures scenario visualization settings.
