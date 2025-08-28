import argparse
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import get_traces
from scipy.stats import gmean

parser = argparse.ArgumentParser()

parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--cpu-model', type=str, required=True)
parser.add_argument('--correlations-type', type=str, required=True)
parser.add_argument('--with-base', action="store_true", required=False)
parser.set_defaults(with_base=False)

args = parser.parse_args()

run_type = args.run_type.split(',')[0]
cpu_model = args.cpu_model.split(',')[0]
correlations_type = args.correlations_type.split(',')[0]

if args.with_base: print("Running with base model")
if correlations_type == "base": run_labelled = False
else: run_labelled = True
label_file_dir = "/work/muke/Branch-Correlations/correlations/"
checkpoint_dir = "/mnt/data/checkpoints-expanded-x86/"
results_dir = "/mnt/data/results/branch-project/gem5-results/"+run_type+"/"+correlations_type+"/"+cpu_model+"/"
base_results_dir = "/mnt/data/results/branch-project/gem5-results/base/base/"+cpu_model+"/"
correlations_dir = "/work/muke/Branch-Correlations/correlations/"

#run checkpoints
processes = []
for bench in get_traces.benchmarks:

    os.chdir(checkpoint_dir+bench)
    run_dir = "/work/muke/spec2017-x86/benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000/"

    if run_labelled:
        while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
        p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/spec_h2p_automation.py --run-type "+run_type+" --correlations-type " +correlations_type+" --cpu-model "+cpu_model, shell=True)
        processes.append(p)

    if args.with_base or not run_labelled:
        while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
        p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/spec_h2p_automation.py --run-type "+run_type+" --correlations-type base --cpu-model "+cpu_model, shell=True)
        processes.append(p)

for p in processes:
    code = p.wait()
    if code is not None and code != 0: print(p.args); exit(1)

#aggregate stats
for bench in get_traces.benchmarks:
    bench_name = bench.split(".")[1].split("_")[0]
    os.chdir(checkpoint_dir+bench)
    for chkpt_dir in os.listdir(os.getcwd()):
        if "bbvs" in chkpt_dir: continue
        run_name = chkpt_dir.split("checkpoints.")[1]
        if len(run_name) != 1 or not run_name.isdigit(): continue #only care about test set
        name = bench_name+"."+run_name
        raw_results_dir = results_dir+"/"+name+"/raw/"
        os.chdir(raw_results_dir)
        p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+bench+" "+run_name, shell=True, check=True)
        subprocess.run("cp results.txt ../", shell=True, check=True)
        subprocess.run("cp accuracies.txt ../", shell=True, check=True)
        if args.with_base:
            raw_results_dir = base_results_dir+"/"+name+"/raw/"
            os.chdir(raw_results_dir)
            p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+bench+" "+run_name, shell=True, check=True)
            subprocess.run("cp results.txt ../", shell=True, check=True)
            subprocess.run("cp accuracies.txt ../", shell=True, check=True)

#generate differences between labelled and base
if not run_labelled: exit(0) #nothing to compare to

prefix = "system.switch_cpus."

stats = {
    "CPI"
}

def get_values(results):
    values = {}
    results = open(results, "r").readlines()
    for line in results:
        name = line.split()[0]
        value = line.split()[1]
        if name in stats:
            values[name] = float(value)
    return values

def get_mpki(accuracy_file):
    simpoint_length = 100e6
    total_mpki = 0
    all_accuracies = []
    stats = {}
    accuracies = open(accuracy_file, "r").read().splitlines()
    for line in accuracies:
        addr, total, incorrect, _ = line.split()
        correct = float(total) - float(incorrect)
        mpki = (float(incorrect) / simpoint_length) * 1000
        accuracy = (correct / float(total)) * 100
        stats[addr+"_mpki"] = mpki
        stats[addr+"_accuracy"] = accuracy
        total_mpki += mpki
        all_accuracies.append(accuracy)
    stats["total_mpki"] = total_mpki
    stats["average_accuracy"] = gmean(all_accuracies)
    return stats

os.chdir(base_results_dir)
base_results = {}
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        base_results[f] = get_values(f+"/results.txt")
        base_results[f].update(get_mpki(f+"/accuracies.txt"))

os.chdir(results_dir)
differences = open("differences", "w")
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        differences.write(f+":\n")
        base_result = base_results[f]
        label_result = get_values(f+"/results.txt")
        label_result.update(get_mpki(f+"/accuracies.txt"))
        differences.write("\tBase CPI: "+str(base_result['CPI'])+"\n")
        differences.write("\tLabel CPI: "+str(label_result['CPI'])+"\n")
        differences.write("\tBase H2P MPKI: "+str(base_result['total_mpki'])+"\n")
        differences.write("\tLabel H2P MPKI: "+str(label_result['total_mpki'])+"\n")
        differences.write("\tBase H2P Accuracy: "+str(base_result['average_accuracy'])+"\n")
        differences.write("\tLabel H2P Accuracy: "+str(label_result['average_accuracy'])+"\n")
        for field in label_result:
            base_value = base_result[field]
            label_value = label_result[field]
            difference = ((label_value - base_value) / base_value) * 100
            if "." in field: field = field.split(".")[-1]
            differences.write("\t"+field+": "+str(difference)+"\n")
        differences.write("\n")

differences.close()
