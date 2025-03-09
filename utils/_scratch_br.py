import polars as pl

df = pl.read_parquet("/mnt/data/results/branch-project/traces/623.xalancbmk_s.train.0.1.trace")

'''
python /work/muke/Branch-Correlations/utils/get_traces.py 623.xalancbmk_s validate
[('623.xalancbmk_s.train.0.1.trace', 0.0239494), ('623.xalancbmk_s.train.0.2.trace', 0.221419), ('623.xalancbmk_s.train.0.3.trace', 0.0275644), ('623.xalancbmk_s.train.0.4.trace', 0.482603), ('623.xalancbmk_s.train.0.5.trace', 0.243109), ('623.xalancbmk_s.train.0.6.trace', 0.000451875), ('623.xalancbmk_s.train.0.7.trace', 0.000903751)]

'''

# mpki: (float(self.incorrect) / self.instructions) * 1000
import polars as pl
import pickle
from typing import List, Tuple, Optional
from utils.get_traces import get_by_workload, trace_dir, benchmarks

def make_mispred_dict(files_weights: Optional[List[Tuple[str, float]]]):    
    mispred_dict = {}
    for file, weight in files_weights:
        df = pl.read_parquet(trace_dir+file)
        for inst_addr in df['inst_addr'].unique():
            filtered = df.filter(df['inst_addr'] == inst_addr)
            #total = filtered.shape[0]
            incorrect = filtered['mispredicted'].sum()
            #correct = total - incorrect
            mispred_dict[inst_addr] = incorrect * weight
    sorted_br = sorted(mispred_dict, key=mispred_dict.get, reverse=True)
    return mispred_dict, sorted_br


for benchmark in benchmarks:
    workload_dict = get_by_workload(benchmark, "validate")

    mispred_dicts = {}
    sorted_brs = {}

    for workload, files_weights in workload_dict.items():
        mispred_dicts[workload], sorted_brs[workload] = make_mispred_dict(files_weights)
        print("total addrs: {}, non-all-correct brs: {}".format(len(mispred_dicts[workload]), len([k for k, v in mispred_dicts[workload].items() if v > 0])))

