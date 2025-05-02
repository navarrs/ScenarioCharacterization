# ScenarioCharacterization

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

3. Compute Trajectory Anomalies
```
cd safeshift_tools
./compute_features primitives training
```