import subprocess
import os
import sys
import time
import psutil

spec_path = "/work/muke/spec2017-x86/"
expanded_spec_path = "/work/muke/spec2017-expanded-x86/"
spec = ["600.perlbench_s", #"605.mcf_s",
       "625.x264_s", "631.deepsjeng_s",
       #"641.leela_s", "657.xz_s",
       "620.omnetpp_s", "602.gcc_s", "623.xalancbmk_s",
       "648.exchange2_s"]
gem5 = "/work/muke/PND-Loads/gem5-gen/"
workloads = "/work/muke/alberta-workloads/"

procs = []

def load_balance():
    time.sleep(60*2)
    while psutil.virtual_memory().percent > 70 or psutil.cpu_percent() > 80: time.sleep(60*5)

def run_test():
    for bench in spec:
        if bench in ["602.gcc_s", "657.xz_s"]: run_dir = expanded_spec_path+"benchspec/CPU/"+bench+"/run/modified/run_peak_refspeed_mytest-64.0000"
        else: run_dir = spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000"
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+"."+str(c)
            p = subprocess.Popen(gem5+"build/X86/gem5.fast --outdir="+checkpoint_path+"/checkpoints."+str(c)+" "+gem5+"configs/deprecated/example/se.py --cpu-type=X86KvmCPU --take-simpoint-checkpoint=/work/muke/simpoints-x86/"+bench_name+".simpts,/work/muke/simpoints-x86/"+bench_name+".weights,100000000,10000000 -c "+command.split()[0]+" --options=\""+' '.join(command.split()[1:])+"\" --mem-size=50GB 2>&1 > "+bench_name+".out 2>&1", shell=True)
            procs.append(p)
            load_balance()

def run_train():
    for bench in spec:
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_train_mytest-64.0000")
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+".train."+str(c)
            p = subprocess.Popen(gem5+"build/X86/gem5.fast --outdir="+checkpoint_path+"/checkpoints.train."+str(c)+" "+gem5+"configs/deprecated/example/se.py --cpu-type=X86KvmCPU --take-simpoint-checkpoint=/work/muke/simpoints-x86/"+bench_name+".simpts,/work/muke/simpoints-x86/"+bench_name+".weights,100000000,10000000 -c "+command.split()[0]+" --options=\""+' '.join(command.split()[1:])+"\" --mem-size=50GB 2>&1 > "+bench_name+".out 2>&1", shell=True)
            procs.append(p)
            load_balance()

def run_alberta():
    for bench in spec:
        if bench == "600.perlbench_s": continue
        checkpoint_path = "/mnt/data/checkpoints-expanded-x86/"+bench
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
        stripped_name = bench.split('.')[1].split('_')[0]
        for workload in os.listdir(workloads+stripped_name):
            subprocess.run("cp -r "+workloads+stripped_name+"/"+workload+"/input/* .", shell=True)
            if bench == "602.gcc_s": binary = "./sgcc_peak.mytest-64"
            else: binary = "./"+bench.split('.')[1]+"_peak.mytest-64"
            control = open("control", "r")
            flags = control.readlines()[0].strip()
            control.close()
            bench_name = bench+"."+workload
            p = subprocess.run(gem5+"build/X86/gem5.fast --outdir="+checkpoint_path+"/checkpoints."+str(workload)+" "+gem5+"configs/deprecated/example/se.py --cpu-type=X86KvmCPU --take-simpoint-checkpoint=/work/muke/simpoints-x86/"+bench_name+".simpts,/work/muke/simpoints-x86/"+bench_name+".weights,100000000,10000000 -c "+binary+" --options=\""+flags+"\" --mem-size=50GB 2>&1 > "+bench_name+".out 2>&1", shell=True, check=True)
            #procs.append(p)
            #load_balance()

run_test()
run_train()
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
