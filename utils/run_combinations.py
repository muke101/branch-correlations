import itertools
import time
import argparse
import os
import subprocess
from collections import defaultdict


parser = argparse.ArgumentParser(prog='run_combinations', description='')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--h2p', type=str, required=True)
parser.add_argument('--branch-file', type=str, required=True)
parser.add_argument('--num-branches', type=int, required=True)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
h2p = args.h2p.split(',')[0]
branch_file = args.branch_file.split(',')[0]
num_branches = args.num_branches

run_dir = "/work/muke/spec2017-x86/benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
spec = "/work/muke/spec2017-x86/"

all_branches = open(branch_file, 'r').read().splitlines()[:num_branches]
combinations = []
for i in range(1,  len(all_branches) + 1):
        combinations.extend(itertools.combinations(all_branches, i))

length_counts = defaultdict(int)
procs = []
for combination in combinations:
    length_counts[len(combination)] += 1
    correlated = open(run_dir+"correlated", "w")
    for branch in combination:
        correlated.write(branch + " 1 \n")
    correlated.write(h2p + " 1 \n")
    correlated.close()
    results_file = "combination_" + str(len(combination)) + "_" + str(length_counts[len(combination)])
    os.chdir(run_dir)
    run = "H2PS=./h2p CORRELATIONS=./correlated /work/muke/Branch-Correlations/gem5-tage/build/X86/gem5.fast /work/muke/Branch-Correlations/gem5-tage/configs/deprecated/example/se.py --cpu-type=DerivO3CPU --caches --restore-with-cpu=AtomicSimpleCPU --restore-simpoint-checkpoint -r 2 --checkpoint-dir=/mnt/data/checkpoints-expanded-x86/641.leela_s/checkpoints.0 --mem-size=50GB -c ./leela_s_peak.mytest-64 --options=\"ref.sgf\" 2> >(grep -e 'PREDICTION' -e 'Warmed up!' | pypy3 /work/muke/Branch-Correlations/utils/record_h2p_accuracies.py "+results_file+")"
    p = subprocess.Popen(run, shell=True, executable="/bin/bash")
    time.sleep(1)
    procs.append(p)
    if len(procs) >= 30:
        for p in procs:
            p.wait()
        procs = []

for p in procs:
    p.wait()
