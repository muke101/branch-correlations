import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import get_traces

chkpt_dir = "/mnt/data/checkpoints-expanded-x86/"

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
for bench in get_traces.branchnet_benchmarks:
    os.chdir(chkpt_dir+bench)
    while psutil.virtual_memory().percent > 60 and psutil.cpu_percent() > 90: time.sleep(60*5)
    p = subprocess.Popen("python3 /work/muke/Branch-Correlations/utils/spec_trace_automation-x86.py", shell=True)#, check=True)
    processes.append(p)

for p in processes:
    code = p.wait()
    if code is not None and code != 0: print(p.args); 
