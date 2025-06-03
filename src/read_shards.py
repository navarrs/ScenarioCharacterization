import pyarrow.dataset as ds
import duckdb

shard_path = "/data/driving/waymo/cache/shards/gt/"
data = ds.dataset(shard_path, format="parquet")

table = data.to_table().to_pandas()

result = duckdb.query(
    """
    SELECT scenario_id
    FROM '/data/driving/waymo/cache/shards/gt/*.parquet'
    WHERE base_score_scenes > 800
    """
).to_df()

print(result)