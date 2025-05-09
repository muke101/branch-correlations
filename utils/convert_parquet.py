import polars as pl
import csv
import sys
import re
import io
import itertools

header = "tick,inst_addr,pred_addr,jump_addr,taken,mispredicted\n"
header_io = io.StringIO(header)

buffer = io.StringIO()
buffer.write(header)

trigger_found = False
for line in sys.stdin:
    if trigger_found:
        buffer.write(line)
    elif "up!" in line:
        trigger_found = True

#def filtered_input(stream):
#    started = False
#    for line in stream:
#        if started:
#            yield line
#        elif "up!" in line:
#            print("Tracing started")
#            started = True
#
#filtered_stream = filtered_input(sys.stdin)

#csv_inp = csv.DictReader(itertools.chain(header_io, filtered_stream))
#csv_inp = csv.DictReader(itertools.chain(header_io, sys.stdin))
#df = pl.DataFrame(list(csv_inp))
#df = df.with_columns([pl.col(col).cast(pl.Int64, strict=False) for col in df.columns])
buffer.seek(0)
df = pl.read_csv(buffer, schema_overrides={
    "tick": pl.Int64,
    "inst_addr": pl.Int64,
    "pred_addr": pl.Int64,
    "jump_addr": pl.Int64,
    "taken": pl.Int64,
    "mispredicted": pl.Int64,
})
df.write_parquet(sys.argv[1])
