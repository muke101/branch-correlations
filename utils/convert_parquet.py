import polars as pd
import csv
import sys
import re

def filter_csv():
    pattern = re.compile(r'^\w+,\w+,\w+,\w+$')
    for line in sys.stdin:
        line = line.strip()
        if pattern.fullmatch(line):
            yield line

csv_inp = csv.DictReader(filter_csv())
df = pd.DataFrame(list(csv_inp))
df = df.with_columns([pd.col(col).cast(pd.Int64, strict=False) for col in df.columns])
df.write_parquet(sys.argv[1])
