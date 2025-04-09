import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time

model = "m4"
chkpt_dir = "/mnt/data/checkpoints-expanded/"
results_dir = "/mnt/data/results/branch-project/big-branchnet/"+cpu_model+"/"
benches = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
           "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "648.exchange2_s"]

if len(sys.argv) > 1:
    sub_benches = []
    for bench in sys.argv[1:]:
        if bench in benches:
            sub_benches.append(bench)
        else:
            print("Unknown benchmark: ", bench)
            exit(1)
    benches = sub_benches


#run checkpoints
for bench in benches:
    os.chdir(chkpt_dir+bench)

    while psutil.virtual_memory().percent > 60 and psutil.cpu_percent() > 90: time.sleep(60*5)
    p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/spec_branchnet_automation.py "+cpu_model, shell=True)
    processes.append(p)

#aggregate stats
for bench in benches:
    name = bench.split(".")[1].split("_")[0]
    
    for i in (0,1,2):
        if os.path.exists(results_dir+name+'.'+str(i)):
            raw_results_dir = results_dir+name+'.'+str(i)+"/raw/"
            os.chdir(raw_results_dir)
            p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+bench+" "+str(i), shell=True)
            p.wait()
            subprocess.Popen("cp results.txt ../", shell=True)
            raw_results_dir = base_results_dir+name+'.'+str(i)+"/raw/"
            os.chdir(raw_results_dir)
            p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/aggregate_stats.py "+bench+" "+str(i), shell=True)
            p.wait()
            subprocess.Popen("cp results.txt ../", shell=True)

#generate differences between labelled and base
if label_file_type == "base": exit(0) #nothing to compare to

prefix = "system.switch_cpus."

#FIXME: new stats for new project

stats = {
    "CPI", 
}
stats_to_diff = {
    "CPI", 
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

os.chdir(base_results_dir)
base_results = {}
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        base_results[f] = get_values(f+"/results.txt")

os.chdir(results_dir)
differences = open("differences", "w")
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        differences.write(f+":\n")
        base_result = base_results[f]
        branchnet_result = get_values(f+"/results.txt")
        differences.write("\tBase CPI: "+str(base_result['CPI'])+"\n")
        differences.write("\tBranchNet CPI: "+str(branchnet_result['CPI'])+"\n")
        for field in branchnet_result:
            if field not in stats_to_diff: continue
            base_value = base_result[field]
            branchnet_value = branchnet_result[field]
            difference = ((branchnet_value - base_value) / base_value) * 100
            if "." in field: field = field.split(".")[-1]
            differences.write("\t"+field+": "+str(difference)+"\n")
        differences.write("\n")

differences.close()
