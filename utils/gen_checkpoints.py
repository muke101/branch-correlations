import subprocess
from subprocess import Popen
import sys
import os
import argparse

spec_path = "/work/muke/spec2017-expanded/"
gem5 = "/work/muke/Branch-Correlations/gem5-gen/"

parser = argparse.ArgumentParser(prog='gen_expanded_checkpoints', description='')

parser.add_argument('--bench', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
args = parser.parse_args()

bench = args.bench.split(',')[0]
run = args.run.split(',')[0]
checkpoint_path = "/mnt/data/checkpoints-expanded/"+bench
if 'train' in run:
    run_number = int(run.split('.')[1])
    work_dir = "modified/run_peak_train_mytest-64.0000"
else:
    run_number = int(run)
    work_dir = "run_peak_refspeed_mytest-64.0000"

os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/"+work_dir)
specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
command = commands[run_number].split('>')[0]
bench_name = bench+"."+run
subprocess.run(gem5+"build/ARM/gem5.fast --outdir="+checkpoint_path+"/checkpoints."+run+" "+gem5+"configs/deprecated/example/se.py --cpu-type=NonCachingSimpleCPU --take-simpoint-checkpoint=/work/muke/simpoints-expanded/"+bench_name+".simpts,/work/muke/simpoints-expanded/"+bench_name+".weights,100000000,10000000 -c "+command.split()[0]+" --options=\""+' '.join(command.split()[1:])+"\" --mem-size=50GB 2>&1 > "+bench_name+".out 2>&1", shell=True, check=True)
