import subprocess
import os
import sys
import time
import psutil
import get_traces

expanded_spec_path = "/work/muke/spec2017-expanded-x86/"
spec_path = "/work/muke/spec2017-x86/"
workloads = "/work/muke/alberta-workloads/"

procs = []

def load_balance():
    while psutil.virtual_memory().percent > 70 or psutil.cpu_percent() > 80: time.sleep(60*5)

def run_test():
    for bench in get_traces.branchnet_benchmarks:
        if bench in ["602.gcc_s", "657.xz_s"]: run_dir = expanded_spec_path+"benchspec/CPU/"+bench+"/run/modified/run_peak_refspeed_mytest-64.0000"
        else: run_dir = spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000"
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+"."+str(c)
            p = subprocess.Popen("valgrind --tool=exp-bbv --bb-out-file="+checkpoint_path+"/bbvs."+str(c)+" "+command.split()[0]+" "+' '.join(command.split()[1:])+" 2>&1 > "+bench_name+".out 2>&1", shell=True)
            procs.append(p)
            load_balance()

def run_train():
    for bench in get_traces.branchnet_benchmarks:
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_train_mytest-64.0000")
        specinvoke = subprocess.run([expanded_spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+".train."+str(c)
            p = subprocess.Popen("valgrind --tool=exp-bbv --bb-out-file="+checkpoint_path+"/bbvs.train."+str(c)+" "+command.split()[0]+" "+' '.join(command.split()[1:])+" 2>&1 > "+bench_name+".out 2>&1", shell=True)
            procs.append(p)
            load_balance()

def run_alberta():
    commands = []
    for bench in get_traces.branchnet_benchmarks:
        if bench == "600.perlbench_s": continue
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
        stripped_name = bench.split('.')[1].split('_')[0]
        for workload in os.listdir(workloads+stripped_name):
            subprocess.run("cp -r "+workloads+stripped_name+"/"+workload+"/input/* .", shell=True)
            if bench == "602.gcc_s": binary = "./sgcc_peak.mytest-64"
            else: binary = "./"+bench.split('.')[1]+"_peak.mytest-64"
            control = open("control", "r")
            lines = control.readlines()
            for c, line in enumerate(lines):
                run_name = workload
                if len(lines) > 1: 
                    run_name += "-" + str(c)
                flags = line.strip()
                bench_name = bench+"."+run_name
                #if len(lines) > 1:
                #    subprocess.run("valgrind --tool=exp-bbv --bb-out-file="+checkpoint_path+"/bbvs."+str(run_name)+" "+binary+" "+flags+" 2>&1 > "+bench_name+".out 2>&1", shell=True, check=True)
                #else:
                p = subprocess.Popen("valgrind --tool=exp-bbv --bb-out-file="+checkpoint_path+"/bbvs."+str(run_name)+" "+binary+" "+flags+" 2>&1 > "+bench_name+".out 2>&1", shell=True)
                procs.append(p)
                load_balance()
            control.close()

#run_test()
#run_train()
run_alberta()

active_procs = procs.copy()

while active_procs:
    for proc in active_procs[:]:
        code = proc.poll()
        if code is not None:
            active_procs.remove(proc)
            if code != 0:
                print("Crash: ", proc.args)
    time.sleep(60*5)
