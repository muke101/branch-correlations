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
parser.add_argument('--branch-files', type=str, required=True)
parser.add_argument('--num-branches', type=int, required=True)
parser.add_argument('--start', type=int, required=False)

args = parser.parse_args()

h2p = args.h2p.split(',')[0]
branch_files = args.branch_files.split(',')
num_branches = args.num_branches
if args.start: start = args.start
else: start = 1

spec = "/work/muke/spec2017-x86/"
h2p_dir = "/work/muke/Branch-Project/h2ps/"

for benchmark in get_traces.benchmarks:
    correlations_dir = "/work/muke/Branch-Correlations/correlations/"+benchmark+"/"
    procs = []
    for branch_file in branch_files:
        if branch_file == "base": 
            while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
            p = subprocess.Popen("python3 run_all_chkpts.py --run-type base --cpu-model m4-0 --correlations-type base", shell=True)
            procs.append(p)
            continue
        all_branches = open(branch_file, 'r').read().splitlines()[:num_branches]
        for i in range(start, num_branches):
            correlation_name = "length_"+str(i)
            os.makedirs(correlations_dir+"/"+branch_file+"/", exist_ok=True)
            correlated = open(correlations_dir+"/"+branch_file+"/"+correlation_name, "w")
            for branch in all_branches[:i]:
                correlated.write(branch + " 1 \n")
            correlated.write(h2p + " 1 \n")
            correlated.close()
            while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
            p = subprocess.Popen("python3 run_all_chkpts.py --run-type "+branch_file+" --cpu-model m4-0 --correlations-type "+correlation_name, shell=True)
            procs.append(p)
    
    for p in procs:
        p.wait()
