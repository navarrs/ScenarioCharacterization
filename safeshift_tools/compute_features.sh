#!/bin/zsh

declare -a shards=(0 1 2 3 4 5 6 7 8 9)

# Get the input arguments
feature="$1"
split="$2"

# Check if both arguments are provided
if [[ -z "$feature" || -z "$split" ]]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 {lanes|frenet|primitives} {training|validation|testing}"
  exit 1
fi

# Check if mode_choice is valid
case "$split" in
  training|validation|testing)
    # valid option, do nothing
    ;;
  *)
    echo "Invalid mode: $split"
    echo "Allowed modes: training, validation, testing"
    exit 1
    ;;
esac

case "$feature" in
  lanes)
   for j in "${shards[@]}"
   do
      echo "Caching closest lanes features for $split split shard $j"
      python compute_closest_lanes.py --split $split --shard_idx ${j} --parallel
      echo "Caching closest lanes features for $split-hist split shard $j"
      python compute_closest_lanes.py --split $split --shard_idx ${j} --parallel --hist_only
   done
    ;;
  frenet)
   for j in "${shards[@]}"
   do
      echo "Caching frenet interpolation features for $split split shard $j"
      python compute_frenet_interp.py --split $split --shard_idx ${j} --parallel
      echo "Caching frenet interpolation for $split-hist split shard $j"
      python compute_frenet_interp.py --split $split --shard_idx ${j} --parallel --hist_only
   done
    ;;
   primitives)
   for j in "${shards[@]}"
   do
      echo "Caching primitives features for full dataset"
      python compute_primitives.py --parallel
      echo "Caching primitives features for hist dataset"
      python compute_primitives.py --parallel --hist_only
      echo "Caching primitives features for extrap dataset"
      python compute_primitives.py --parallel --extrap
   done
    ;;
  *)
    echo "Invalid option: $feature"
    echo "Usage: $0 {lanes|frenet|primitives} {training|validation|testing}"
    exit 1
    ;;
esac

echo "...done."