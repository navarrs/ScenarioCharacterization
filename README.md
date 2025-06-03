# ScenarioCharacterization

## Running a feature processor
```
uv run src/run_processor.py processor=features characterizer=feature
```

## Running a score processor
```
uv run src/run_processor.py processor=scores characterizer=score
```

### OLD STUFF
Re-spliting the data:
```
cd data_tools
python resplit.py --base_path /data/driving/waymo/
```




Refactored process to get all SafeShift features:

1. Compute Closest Lanes:
```
cd safeshift_tools
chmod +x compute_features.sh
./compute_features lanes testing
./compute_features lanes validation
./compute_features lanes training
```

2. Compute Frenet Interpolations
```
cd safeshift_tools
./compute_features frenet testing
./compute_features frenet validation
./compute_features frenet training
```

3. Compute Trajectory Anomalies (here the split key doesn't matter tho)
```
cd safeshift_tools
./compute_features primitives training
```

4. Compute Map Features
```
cd safeshift_tools
./compute_features map training
./compute_features map validation
./compute_features map testing
```

5. Compute Conflict Points
```
cd safeshift_tools
./compute_features conflictpoints training
./compute_features conflictpoints validation
./compute_features conflictpoints testing
```