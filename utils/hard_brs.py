import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)

import pickle
from utils.get_traces import benchmarks

hard_branches_dir = "/mnt/data/results/branch-project/h2ps/validate/"
stats_dir = "/work/muke/Branch-Correlations/stats/"

NUM_BRS_TO_PRINT = 100

def greedy_select_top_brs(list_inputs, mpki_dicts, sorted_brs, num_brs):
    selected_brs = []
    next_br_idx = [0] * len(list_inputs)
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
                    print("KeyError: [{}] br: {} not in workload: {}".format(i, br, inppp))
            next_br_total_mpki[j] = total_mpki
            next_br_pc[j] = br
        max_j = next_br_total_mpki.index(max(next_br_total_mpki))
        selected_brs.append(next_br_pc[max_j])
    return selected_brs


for benchmark in benchmarks:
    with open("{}mispred_dicts_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
        mispred_dicts = pickle.load(f)
    with open("{}sorted_brs_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
        sorted_brs = pickle.load(f)
    workloads = list(mispred_dicts.keys())
    brs = greedy_select_top_brs(workloads, mispred_dicts, sorted_brs, NUM_BRS_TO_PRINT)

    with open(hard_branches_dir+benchmark, 'w') as f:
        for br in brs:
            f.write(hex(br)+'\n')
    
    print("finished writing hard branches for {}".format(benchmark))

