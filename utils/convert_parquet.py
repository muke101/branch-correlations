import polars as pd
import csv
import sys
import re
import io
import itertools

header = "tick,inst_addr,pred_addr,jump_addr,taken,mispredicted\n"
header_io = io.StringIO(header)

csv_inp = csv.DictReader(itertools.chain(header_io, sys.stdin))
df = pd.DataFrame(list(csv_inp))
df = df.with_columns([pd.col(col).cast(pd.Int64, strict=False) for col in df.columns])
df.write_parquet(sys.argv[1])
