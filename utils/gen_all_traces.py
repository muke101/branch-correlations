import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time

chkpt_dir = "/mnt/data/checkpoints-expanded/"
results_dir = "/mnt/data/results/branch-project/traces/"
benches = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
           "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "648.exchange2_s"]

if len(sys.argv) > 1:
    sub_benches = []
    for bench in sys.argv[5:]:
        if bench in benches:
            sub_benches.append(bench)
        else:
            print("Unknown benchmark: ", bench)
            exit(1)
    benches = sub_benches

#run checkpoints
processes = []
for bench in benches:
    os.chdir(chkpt_dir+bench)
    while psutil.virtual_memory().percent > 60 and psutil.cpu_percent() > 90: time.sleep(60*5)
    p = subprocess.run("python3 /work/muke/Branch-Correlations/utils/spec_trace_automation.py", shell=True, check=True)
    #processes.append(p)

for p in processes:
    code = p.wait()
    if code is not None and code != 0: print(p.args); exit(1)
