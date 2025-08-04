import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)

import polars as pl
import pickle
from collections import defaultdict
from typing import List, Tuple, Optional
from utils.get_traces import get_by_workload, trace_dir, benchmarks

def make_mispred_dict(files_weights: Optional[List[Tuple[str, float]]]):    
    mispred_dict = defaultdict(int)
    denom_dict = defaultdict(int)
    for file, weight in files_weights:
        df = pl.read_parquet(trace_dir+file)
        for inst_addr in df['inst_addr'].unique():
            filtered = df.filter((df['inst_addr'] == inst_addr) & (df['warmed_up'] == 1))
            total = filtered.shape[0]
            if total == 0: continue
            incorrect = filtered['mispredicted'].sum()
            #correct = total - incorrect
            mispred_dict[inst_addr] += incorrect * weight
            denom_dict[inst_addr] += weight
    for k in mispred_dict.keys():
        mispred_dict[k] /= denom_dict[k]
    sorted_br = sorted(mispred_dict, key=mispred_dict.get, reverse=True)
    return mispred_dict, sorted_br

def process_benchmark(benchmark):
    workload_dict = get_by_workload(benchmark, "validate")

    mispred_dicts = {}
    sorted_brs = {}
    print("benchmark: {}".format(benchmark))
    if os.path.exists("sorted_brs_{}.pkl".format(benchmark)): return
    for workload, files_weights in workload_dict.items():
        mispred_dicts[workload], sorted_brs[workload] = make_mispred_dict(files_weights)
        print("workload: {}, total addrs: {}, non-all-correct brs: {}".format(workload, len(mispred_dicts[workload]), len([k for k, v in mispred_dicts[workload].items() if v > 0])))

    with open("mispred_dicts_{}.pkl".format(benchmark), 'wb') as f:
        pickle.dump(mispred_dicts, f)

    with open("sorted_brs_{}.pkl".format(benchmark), 'wb') as f:
        pickle.dump(sorted_brs, f)

if __name__ == "__main__":
    num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_benchmark, bench): bench for bench in ["641.leela_s"]}

        for future in as_completed(futures):
            trace = futures[future]
            try:
                future.result()  # Raises exception if one occurred
            except Exception as e:
                print(f"Error processing trace {trace}: {e}")
