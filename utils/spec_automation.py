import os
import sys
import subprocess
from subprocess import Popen
import psutil
import random
import time

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
}

#run from base spec dir
base_dir = os.getcwd() # = /work/muke/checkpoints/benchmark
base_run = False
run_type = sys.argv[1]
addr_file_type = sys.argv[2]
if addr_file_type == "base": base_run = True
cpu_model = sys.argv[3]
spec_path = "/work/muke/spec2017/"
gem5 = "/work/muke/PND-Loads/gem5/"
results_dir = "/work/muke/results/"+run_type+"/"+addr_file_type+"/"+cpu_model+"/"
addr_file_dir = "/work/muke/PND-Loads/addr_files/"
benchmark = base_dir.split("/")[4]
run_dir = spec_path+"benchspec/CPU/"+benchmark+"/run/run_peak_refspeed_mytest-64.0000/"
os.chdir(run_dir)
specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if not line.startswith(b"#")]
os.chdir(base_dir)
random.seed(sum(ord(c) for c in base_dir))
procs = []

#iterate over all checkpoint.n dirs
for out_dir in os.listdir(base_dir):
    run_number = out_dir[-1]
    command = commands[int(run_number)]
    command = command.split('>')[0]
    cpt_number = 0
    out_dir = os.path.join(base_dir,out_dir)
    #iterate over checkpoint.n
    for chkpt_dir in os.listdir(out_dir):
        #find cpt.m dir
        if os.path.isdir(os.path.join(out_dir, chkpt_dir)) and chkpt_dir.startswith('cpt.'):
            waited = 0
            finished = False
            cpt_number += 1
            binary = run_dir+command.split()[0]
            benchmark_name = benchmark.split("_")[0].split(".")[1]
            if base_run:
                addr_file = "/work/muke/empty"
            else:
                addr_file = addr_file_dir+addr_file_type+"/"+benchmark
            outdir = results_dir+benchmark_name+"."+run_number+"/raw/"
            if not os.path.exists(outdir): os.makedirs(outdir) #create the parent directories for gem5 stats dir if needed
            outdir += str(cpt_number)+".out"
            run = "ADDR_FILE="+addr_file+" "+gem5+"build/ARM/gem5.fast --outdir="+outdir+" "+gem5+"configs/deprecated/example/se.py --cpu-type=DerivO3CPU --caches --l2cache --restore-simpoint-checkpoint -r "+str(cpt_number)+" --checkpoint-dir "+out_dir+" --restore-with-cpu=AtomicSimpleCPU --mem-size=50GB -c "+binary+" --options=\""+' '.join(command.split()[1:])+"\""
            run += cache_sizes[cpu_model]
            os.chdir(run_dir)
            while psutil.virtual_memory().percent > 60 and psutil.cpu_percent() > 90: time.sleep(60*5)
            p = Popen(run, shell=True)
            os.chdir(base_dir)
            procs.append(p)
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
