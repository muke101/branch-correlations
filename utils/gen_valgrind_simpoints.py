import os
import subprocess
from subprocess import Popen
import get_traces

expanded_spec_path = "/work/muke/spec2017-expanded-x86/"
spec_path = "/work/muke/spec2017-x86/"
simpoint = "/work/muke/SimPoint/bin/simpoint"
workloads = "/work/muke/alberta-workloads/"
procs = []
bench_names = []

def run_test():
    for bench in get_traces.branchnet_benchmarks:
        if bench in ["602.gcc_s", "657.xz_s"]: run_dir = expanded_spec_path+"benchspec/CPU/"+bench+"/run/modified/run_peak_refspeed_mytest-64.0000"
        else: run_dir = spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000"
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = (bench,str(c))
            bench_names.append(bench_name)

def run_train():
    for bench in get_traces.branchnet_benchmarks:
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_train_mytest-64.0000")
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = (bench,"train."+str(c))
            bench_names.append(bench_name)

def run_alberta():
    for bench in get_traces.branchnet_benchmarks:
        if bench == "600.perlbench_s": continue
        os.chdir(expanded_spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
        stripped_name = bench.split('.')[1].split('_')[0]
        for workload in os.listdir(workloads+stripped_name):
            subprocess.run("cp -r "+workloads+stripped_name+"/"+workload+"/input/* .", shell=True)
            if bench == "602.gcc_s": binary = "sgcc_peak.mytest-64"
            else: binary = bench.split('.')[1]+"_peak.mytest-64"
            control = open("control", "r")
            lines = control.readlines()
            for c, line in enumerate(lines):
                run_name = workload
                if len(lines) > 1:
                    run_name += "-" + str(c) 
                bench_name = (bench,run_name)
                bench_names.append(bench_name)
            control.close()

run_test()
run_train()
run_alberta()

for bench,workload in bench_names:
    checkpoint_dir = "/mnt/data/checkpoints-expanded-x86/"
    subprocess.run([simpoint, "-loadFVFile",  checkpoint_dir+bench+"/bbvs."+workload,
            "-k", "search", "-maxK", "10", "-saveSimpoints",
            "/work/muke/simpoints-x86/"+bench+"."+workload+".simpts", "-saveSimpointWeights",
            "/work/muke/simpoints-x86/"+bench+"."+workload+".weights"], check=True)
