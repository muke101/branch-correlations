import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)

import pickle
import polars as pl
from utils.get_traces import benchmarks
import utils.get_traces
import utils.hard_brs
import csv
import itertools
import io
from scipy.stats import gmean
import re
from collections import defaultdict

hard_branches_dir = "/mnt/data/results/branch-project/h2ps/validate/"
stats_dir = "/work/muke/Branch-Correlations/stats/"
branchnet_results_dir = "/mnt/data/results/branch-project/branchnet-results/main/"

def tage_mispreds(benchmark):
    #FIXME: msipred_dicts needs to be generated from the test set, probably want to just grab the trace files and iterate here
    with open("{}mispred_dicts_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
        mispred_dicts = pickle.load(f)
    with open("{}sorted_brs_{}.pkl".format(stats_dir, benchmark), 'rb') as f:
        sorted_brs = pickle.load(f)
    workloads = list(mispred_dicts.keys())
    brs, exceptions = utils.hard_brs.greedy_select_top_brs(workloads, mispred_dicts, sorted_brs, utils.hard_brs.NUM_BRS_TO_PRINT)

    return {b:mispred_dicts[b] for b in mispred_dicts if b in brs}

def branchnet_mispreds(benchmark):
    header = "simpoint,accuracy,correct,total"
    header_io = io.StringIO(header)

    accuracy_dict = {}
    for results_file in os.listdir(branchnet_results_dir+benchmark+"/results/"):
        if results_file.split('.')[1] != 'csv': continue
        br = int(results_file.split('.')[0],16)
        reader = csv.DictReader(itertools.chain(header_io, open(results_file)))
        correct = 0
        total = 0
        for run in reader:
            if run['accuracy'] == 0: continue
            indx = [match.end() for match in re.finditer(re.escape(benchmark), run['simpoint'])][0]
            checkpoint = run['simpoint'][indx:].split('.')[-1]
            workload = run['simpoint'][indx:].split('.')[1]
            #TODO: do we need to be careful about weighting simpoints across multiple workloads?
            if len(workload) > 1 and not workload.isdigit(): continue #only select test set
            weight = utils.get_traces.get_simpoint_weight(benchmark, workload, checkpoint)
            correct += float(run['correct']) * weight
            total += float(run['total']) * weight
        accuracy_dict[br] = correct/total

    return accuracy_dict

if __name__ == "__main__":
    for benchmark in benchmarks:
        branchnet = branchnet_mispreds(benchmark)
        tage = tage_mispreds(benchmark)
        improved = {br for br in branchnet if branchnet[br] > tage[br]}
        print(benchmark+": ",len(improved))
