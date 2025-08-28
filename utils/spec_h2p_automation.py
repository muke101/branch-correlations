import os
import sys
import subprocess
import psutil
import random
import time
import argparse

cache_sizes = {
    "default": "",
    "scaled": " --l1d_size=128KiB --l1i_size=128KiB --l2_size=2MB",
    "very_scaled": " --l1d_size=256KiB --l1i_size=256KiB --l2_size=4MB",
    "x925": " --l1d_size=64KiB --l1i_size=64KiB --l2_size=4MB",
    "a725": " --l1d_size=64KiB --l1i_size=64KiB --l2_size=1MB",
    "a14": " --l1d_size=64KiB --l1i_size=128KiB --l2_size=4MB",
    "a14-tournament": " --l1d_size=64KiB --l1i_size=128KiB --l2_size=4MB",
    "a14-small-mdp": " --l1d_size=64KiB --l1i_size=128KiB --l2_size=4MB",
    "m4": " --l1d_size=128KiB --l1i_size=256KiB --l2_size=16MB",
    "m4-0": " --l1d_size=128KiB --l1i_size=256KiB --l2_size=16MB",
    "m4-small-phast": " --l1d_size=128KiB --l1i_size=256KiB --l2_size=16MB",
}

parser = argparse.ArgumentParser()

parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--cpu-model', type=str, required=True)
parser.add_argument('--correlations-type', type=str, required=True)

args = parser.parse_args()

run_type = args.run_type.split(',')[0]
cpu_model = args.cpu_model.split(',')[0]
correlations_type = args.correlations_type.split(',')[0]

#run from base spec dir
base_dir = os.getcwd() # = /mnt/data/checkpoints-expanded-x86/benchmark
spec_path = "/work/muke/spec2017-x86/"
expanded_spec_path = "/work/muke/spec2017-expanded-x86/"
gem5 = "/work/muke/Branch-Correlations/gem5-tage/"
results_dir = "/mnt/data/results/branch-project/gem5-results/"+run_type+"/"+correlations_type+"/"+cpu_model+"/"
workloads = "/work/muke/alberta-workloads/"
correlations_dir = "/work/muke/Branch-Correlations/correlations/"
h2p_dir = "/work/muke/Branch-Correlations/h2ps/"
benchmark = base_dir.split("/")[4]
random.seed(sum(ord(c) for c in base_dir))
procs = []

def get_bench_flags(run_name):
    commands = []
    #test
    if run_name.isdigit() and len(run_name) == 1:
        if benchmark in ["602.gcc_s", "657.xz_s"]:
            run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/modified/run_peak_refspeed_mytest-64.0000/"
        else:
            run_dir = spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        runs = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if not line.startswith(b"#")]
        os.chdir(base_dir)
        command = runs[int(run_name)]
        command = command.split('>')[0]
        commands.append(command)
        return (run_dir,commands)
    return (None, [])
    ##train
    #elif len(run_name.split('train.')) > 1 and run_name.split('train.')[1].isdigit():
    #    run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_train_mytest-64.0000/"
    #    os.chdir(run_dir)
    #    specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
    #    commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if not line.startswith(b"#")]
    #    os.chdir(base_dir)
    #    command = commands[int(run_name.split('train.')[1])]
    #    command = command.split('>')[0]
    #    commands.append(command)
    ##alberta
    #else:
    #    run_dir = expanded_spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
    #    os.chdir(run_dir)
    #    stripped_name = benchmark.split('.')[1].split('_')[0]
    #    if run_name[-2] == '-' and run_name[-1].isdigit():
    #        run_name = run_name[:-2]
    #    subprocess.run("cp -r "+workloads+stripped_name+"/"+run_name+"/input/* .", shell=True)
    #    if benchmark == "602.gcc_s": binary = "sgcc_peak.mytest-64"
    #    else: binary = benchmark.split('.')[1]+"_peak.mytest-64"
    #    control = open("control", "r")
    #    for line in control.readlines():
    #        flags = line.strip()
    #        command = binary+" "+flags
    #        commands.append(command)
    #    control.close()
    #    os.chdir(base_dir)

#iterate over all checkpoint.n dirs
for chkpt_dir in os.listdir(base_dir):
    if "bbvs" in chkpt_dir: continue
    run_name = chkpt_dir.split("checkpoints.")[1]
    run_dir, commands = get_bench_flags(run_name)
    for c, command in enumerate(commands):
        cpt_number = 0
        out_dir = os.path.join(base_dir,chkpt_dir)
        #iterate over checkpoint.n
        for cpt_dir in sorted(os.listdir(out_dir)):
            #find cpt.m dir
            if os.path.isdir(os.path.join(out_dir, cpt_dir)) and cpt_dir.startswith('cpt.'):
                waited = 0
                finished = False
                cpt_number += 1
                binary = "./"+command.split()[0]
                benchmark_name = benchmark.split("_")[0].split(".")[1]
                outdir = results_dir+benchmark_name+"."+run_name+"/raw/"
                if not os.path.exists(outdir): os.makedirs(outdir) #create the parent directories for gem5 stats dir if needed
                outdir += str(cpt_number)+".out"
                accuracies_file = outdir+"/accuracies"
                h2p_file = h2p_dir+benchmark
                correlations_file = correlations_dir+benchmark+"/"+run_type+"/"+correlations_type
                run = "H2PS="+h2p_file+" CORRELATIONS="+correlations_file+" "+gem5+"build/X86/gem5.fast --outdir="+outdir+" "+gem5+"configs/deprecated/example/se.py --cpu-type=DerivO3CPU --caches --l2cache --restore-with-cpu=AtomicSimpleCPU --restore-simpoint-checkpoint -r "+str(cpt_number)+" --checkpoint-dir="+out_dir+" --mem-size=50GB -c "+binary+" --options=\""+' '.join(command.split()[1:])+"\" "+cache_sizes[cpu_model]+" 1>&2 2> >(grep -e 'PREDICTION' -e 'Warmed up!' | python3 /work/muke/Branch-Correlations/utils/record_h2p_accuracies.py "+accuracies_file+")"
                os.chdir(run_dir)
                while psutil.virtual_memory().percent > 60 or psutil.cpu_percent() > 90: time.sleep(60)
                p = subprocess.Popen(run, shell=True, executable='/bin/bash')
                procs.append(p)
                time.sleep(60)
                os.chdir(base_dir)


for p in procs:
    code = p.wait()
    if code is not None and code != 0: print("Crash: ", p.args); 
