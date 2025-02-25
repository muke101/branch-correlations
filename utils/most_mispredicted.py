import polars as pl
import get_traces
from collections import defaultdict
import sys

benches = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
           "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "648.exchange2_s"]

hard_branches_dir = "/mnt/data/results/branch-project/h2ps/"

n = 100

def get_most_mispredicted(trace_path, n):

    df = pl.read_parquet(get_traces.trace_dir+trace_path)
    pc_col = df.columns[1]
    mispred_col = df.columns[-1]
    df = df.with_columns(pl.col(mispred_col).cast(pl.Boolean))

    mispred_counts = (
        df.filter(pl.col(mispred_col))
        .group_by(pc_col)
        .agg(pl.len().alias("mispred_count"))
        .sort("mispred_count", descending=True)
        .head(n)
    )
    count_df = df["inst_addr"].value_counts()

    mispreds = []
    for row in mispred_counts.iter_rows():
        num_executions = count_df.filter(count_df["inst_addr"] == row[0])['count'][0]
        mispreds.append((row[0], num_executions, row[1]))
    return mispreds

def main():
    set_types = ['validate', 'test']
    for set_type in set_types:
        print("Processing set: ", set_type)
        for bench in benches:
            print("Processing bench: ", bench)
            traces = get_traces.get_trace_set(bench, set_type)
            executions_map = defaultdict(int)
            mispredicted_map = defaultdict(int)
            misprediction_rates = {}

            for trace, weight in traces:
                most_mispredicted = get_most_mispredicted(trace, n)
                for pc, total, incorrect in most_mispredicted:
                    executions_map[pc] += total*weight
                    mispredicted_map[pc] += incorrect*weight
                for pc in executions_map:
                    misprediction_rates[pc] = mispredicted_map[pc]/executions_map[pc]
            hard_branches = [b for b, _ in sorted(misprediction_rates.items(), key=lambda x: x[1], reverse=True)[:100]]

            hard_branches_file = open(hard_branches_dir+set_type+"/"+bench, "w")
            for branch in hard_branches:
                hard_branches_file.write(hex(branch)+"\n")
            hard_branches_file.close()

if __name__ == "__main__":
    main()
