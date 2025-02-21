import polars as pl
import sys

# Load the parquet file
file_path = sys.argv[1]  # Pass the Parquet file path as a command-line argument
n = int(sys.argv[2])
df = pl.read_parquet(file_path)

# Assuming the second field is the instruction PC and the last field is the misprediction bool
pc_col = df.columns[1]  # Second column
mispred_col = df.columns[-1]  # Last column

df = df.with_columns(pl.col(mispred_col).cast(pl.Boolean))

# Aggregate misprediction counts by PC
mispred_counts = (
    df.filter(pl.col(mispred_col))
    .group_by(pc_col)
    .agg(pl.len().alias("mispred_count"))
    .sort("mispred_count", descending=True)
    .head(n)
)

# Get the branch PC with the most mispredictions
#most_mispredicted = mispred_counts.select(pc_col)

for row in mispred_counts.iter_rows():
    #print(f"PC: {hex(row[0])}, Mispredictions: {row[1]}")
    print(f"{hex(row[0])}")
