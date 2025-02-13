import polars as pd
import csv
import sys

csv_inp = csv.DictReader(sys.stdin)
df = pd.DataFrame(list(csv_inp))
df = df.with_columns([pd.col(col).cast(pd.Int64, strict=False) for col in df.columns])
df.write_parquet(sys.argv[1])
