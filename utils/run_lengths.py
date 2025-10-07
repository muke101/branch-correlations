import psutil
import itertools
import time
import argparse
import os
import subprocess
from collections import defaultdict
import get_traces

parser = argparse.ArgumentParser(prog='run_combinations', description='')

parser.add_argument('--h2p', type=str, required=True)
parser.add_argument('--correlation-types', type=str, required=True)
parser.add_argument('--num-branches', type=int, required=True)
parser.add_argument('--start', type=int, required=False)
parser.add_argument('--benchmarks', type=str, required=False)

args = parser.parse_args()

h2p = args.h2p.split(',')[0]
correlation_types = args.correlation_types.split(',')
num_branches = args.num_branches
if args.start: start = args.start
else: start = 1
if args.benchmarks:
    benchmarks = args.benchmarks.split(',')
else:
    benchmarks = get_traces.benchmarks

for benchmark in benchmarks:
    correlations_dir = "/work/muke/Branch-Correlations/correlations/"+benchmark+"/"
    procs = []
    if "base" in correlation_types:
        while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
        p = subprocess.Popen("python3 run_all_train_chkpts.py --run-type base --cpu-model m4-0 --correlations-type base --benchmark "+benchmark, shell=True)
        procs.append(p)

    correlation_types = [c for c in correlation_types if c != "base"]

    for correlation_type in correlation_types:
        all_branches = open(correlations_dir+"/"+correlation_type+"/full", 'r').read().splitlines()[:num_branches]
        all_branches = [i.split()[0] for i in all_branches]
        for i in range(start, num_branches):
            correlation_name = "length_"+str(i)
            #os.makedirs(correlations_dir+"/"+branch_file+"/", exist_ok=True)
            correlated = open(correlations_dir+"/"+correlation_type+"/"+correlation_name, "w")
            for branch in all_branches[:i]:
                correlated.write(branch + " 1 \n")
            correlated.write(h2p + " 1 \n")
            correlated.close()
            while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
            p = subprocess.Popen("python3 run_all_train_chkpts.py --run-type "+correlation_type+" --cpu-model m4-0 --correlations-type "+correlation_name+" --benchmark "+benchmark, shell=True)
            procs.append(p)
    
    for p in procs:
        p.wait()
