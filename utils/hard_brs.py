import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)

import pickle
import polars as pl
from utils.get_traces import benchmarks

hard_branches_dir = "/mnt/data/results/branch-project/h2ps/validate/"
stats_dir = "/work/muke/Branch-Correlations/stats/"

NUM_BRS_TO_PRINT = 100

def greedy_select_top_brs(list_inputs, mpki_dicts, sorted_brs, num_brs):
    selected_brs = []
    next_br_idx = [0] * len(list_inputs)
    exceptions = []
    for i in range(num_brs):
        next_br_total_mpki = [0.0] * len(list_inputs)
        next_br_pc = [0] * len(list_inputs)
        for j, inp in enumerate(list_inputs):
            while sorted_brs[inp][next_br_idx[j]] in selected_brs:
                next_br_idx[j] += 1
            br = sorted_brs[inp][next_br_idx[j]]
            total_mpki = 0
            for inppp in list_inputs:
                try:
                    total_mpki += mpki_dicts[inppp][br]
                except KeyError:
                    exceptions.append({'br': br, 'i': i, 'workload': inppp, 'selected': False})
            next_br_total_mpki[j] = total_mpki
            next_br_pc[j] = br
        max_j = next_br_total_mpki.index(max(next_br_total_mpki))
        selected_brs.append(next_br_pc[max_j])

    for exception in exceptions:
        ii = exception['i']
        if exception['br'] == selected_brs[ii]:
            exception['selected'] = True

    return selected_brs, exceptions


if __name__ == "__main__":
    for benchmark in benchmarks:
        with open("{}mispred_dicts_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
            mispred_dicts = pickle.load(f)
        with open("{}sorted_brs_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
            sorted_brs = pickle.load(f)
        workloads = list(mispred_dicts.keys())
        brs, exceptions = greedy_select_top_brs(workloads, mispred_dicts, sorted_brs, NUM_BRS_TO_PRINT)
    
        with open(hard_branches_dir+benchmark, 'w') as f:
            for br in brs:
                f.write('{}\n'.format(br))
    
        pl.DataFrame(exceptions).write_csv(hard_branches_dir+'exceptions/'+benchmark+".csv")
        
        print("finished writing hard branches for {}".format(benchmark))
