import os
import sys
import subprocess
from subprocess import Popen
import psutil
import random
import time

#run from base spec dir
base_dir = os.getcwd() # = /mnt/data/checkpoints-expanded/benchmark
base_run = False
spec_path = "/work/muke/spec2017/"
expanded_spec_path = "/work/muke/spec2017-expanded/"
gem5 = "/work/muke/Branch-Correlations/trace-gem5/"
results_dir = "/mnt/data/results/branch-project/traces/"
label_file_dir = "/work/muke/Branch-Correlations/label_files/"
workloads = "/work/muke/alberta-workloads/"
benchmark = base_dir.split("/")[4]
random.seed(sum(ord(c) for c in base_dir))
procs = []

def get_bench_flags(run_name):
    #test
    if run_name.isdigit() and len(run_name) == 1:
        if benchmark in ["602.gcc_s", "657.xs_s", "648.exchange2_s"]:
            run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/modified/run_peak_refspeed_mytest-64.0000/"
        else:
            run_dir = spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if not line.startswith(b"#")]
        os.chdir(base_dir)
        command = commands[int(run_name)]
        command = command.split('>')[0]
        command += " 2> /dev/null"
    #train
    elif len(run_name.split('train.')) > 1 and run_name.split('train.')[1].isdigit():
        run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_train_mytest-64.0000/"
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if not line.startswith(b"#")]
        os.chdir(base_dir)
        command = commands[int(run_name.split('train.')[1])]
        command = command.split('>')[0]
        command += " 2> /dev/null"
    #alberta
    else:
        run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
        os.chdir(run_dir)
        stripped_name = benchmark.split('.')[1].split('_')[0]
        subprocess.run("cp -r "+workloads+stripped_name+"/"+run_name+"/input/* .", shell=True)
        if benchmark == "602.gcc_s": binary = run_dir+"sgcc_peak.mytest-64"
        else: binary = benchmark.split('.')[1]+"_peak.mytest-64"
        control = open("control", "r")
        flags = control.readlines()[0].strip()
        control.close()
        os.chdir(base_dir)
        command = binary+" "+flags
        command += " 2> /dev/null"
    return (run_dir,command)

#iterate over all checkpoint.n dirs
for out_dir in os.listdir(base_dir):
    run_name = out_dir.split("checkpoints.")[1]
    print(benchmark, run_name)
    run_dir, command = get_bench_flags(run_name)
    cpt_number = 0
    out_dir = os.path.join(base_dir,out_dir)
    #iterate over checkpoint.n
    for chkpt_dir in os.listdir(out_dir):
        #find cpt.m dir
        if os.path.isdir(os.path.join(out_dir, chkpt_dir)) and chkpt_dir.startswith('cpt.'):
            waited = 0
            finished = False
            cpt_number += 1
            binary = "./"+command.split()[0]
            benchmark_name = benchmark.split("_")[0].split(".")[1]
            outdir = results_dir+benchmark_name+"."+run_name+"/raw/"
            if not os.path.exists(outdir): os.makedirs(outdir) #create the parent directories for gem5 stats dir if needed
            outdir += str(cpt_number)+".out"
            run = gem5+"build/ARM/gem5.fast --outdir=/dev/null "+gem5+"configs/deprecated/example/se.py --cpu-type=DerivO3CPU --caches --restore-simpoint-checkpoint -r "+str(cpt_number)+" --checkpoint-dir "+out_dir+" --restore-with-cpu=NonCachingSimpleCPU --mem-size=50GB -c "+binary+" --options=\""+' '.join(command.split()[1:])+"\" 2> >(tail -n +7 | python3 /work/muke/Branch-Correlations/utils/convert_parquet.py "+results_dir+benchmark+"."+run_name+".trace)"
            os.chdir(run_dir)
            while psutil.virtual_memory().percent > 60 and psutil.cpu_percent() > 90: time.sleep(60*5)
            print(run)
            print()
            #p = Popen(run, shell=True)
            os.chdir(base_dir)
            #procs.append(p)
            continue
            while waited < 60*2 and finished == False:
                time.sleep(10)
                waited += 10
                if Popen.poll(p) != None:
                    finished = True
                    if Popen.wait(p) != 0: print(p.args); exit(1)
            time.sleep(random.uniform(0,1)*60)
            if psutil.virtual_memory().percent < 60 and psutil.cpu_percent() < 90: continue
            if Popen.wait(p) != 0: print(p.args); exit(1)

for p in procs:
    code = Popen.wait(p)
    if code is not None and code != 0: print(p.args); exit(1)
