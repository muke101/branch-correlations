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
parser.add_argument('--benchmark', type=str, required=True)
parser.set_defaults(with_base=False)

args = parser.parse_args()

run_type = args.run_type.split(',')[0]
cpu_model = args.cpu_model.split(',')[0]
correlations_type = args.correlations_type.split(',')[0]
benchmark = args.benchmark.split(',')[0]

if args.with_base: print("Running with base model")
if correlations_type == "base": run_labelled = False
else: run_labelled = True
checkpoint_dir = "/mnt/data/checkpoints-expanded-x86/"
results_dir = "/mnt/data/results/branch-project/gem5-results/"+benchmark+"/"+run_type+"-train/"+correlations_type+"/"+cpu_model+"/"
base_results_dir = "/mnt/data/results/branch-project/gem5-results/"+benchmark+"/base-train/base/"+cpu_model+"/"
if not os.path.exists(results_dir): os.makedirs(results_dir)
if not os.path.exists(base_results_dir): os.makedirs(base_results_dir)
correlations_dir = "/work/muke/Branch-Correlations/correlations/"

#run checkpoints
processes = []
os.chdir(checkpoint_dir+benchmark)
run_dir = "/work/muke/spec2017-expanded-x86/benchspec/CPU/"+benchmark+"/run/run_peak_train_mytest-64.0000/"

if run_labelled:
    while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
    p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/spec_h2p_train_automation.py --run-type "+run_type+" --correlations-type " +correlations_type+" --cpu-model "+cpu_model, shell=True)
    processes.append(p)

if args.with_base or not run_labelled:
    while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60*5)
    p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/spec_h2p_train_automation.py --run-type "+run_type+" --correlations-type base --cpu-model "+cpu_model, shell=True)
    processes.append(p)

for p in processes:
    code = p.wait()
if code is not None and code != 0: print(p.args); exit(1)

#aggregate stats
benchmark_name = benchmark.split(".")[1].split("_")[0]
os.chdir(checkpoint_dir+benchmark)
for chkpt_dir in os.listdir(os.getcwd()):
    if "bbvs" in chkpt_dir: continue
    run_name = chkpt_dir.split("checkpoints.")[1]
    if 'train' not in run_name: continue #only care about train set
    name = benchmark_name+"."+run_name
    raw_results_dir = results_dir+"/"+name+"/raw/"
    if not os.path.exists(raw_results_dir): os.makedirs(raw_results_dir)
    os.chdir(raw_results_dir)
    p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+benchmark+" "+run_name, shell=True, check=True)
    subprocess.run("cp results.txt ../", shell=True, check=True)
    subprocess.run("cp accuracies.txt ../", shell=True, check=True)
    if args.with_base:
        raw_results_dir = base_results_dir+"/"+name+"/raw/"
        os.chdir(raw_results_dir)
        p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+benchmark+" "+run_name, shell=True, check=True)
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
