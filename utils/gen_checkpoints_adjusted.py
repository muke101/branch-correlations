import subprocess
from subprocess import Popen
import sys
import os

bench = sys.argv[1]
run = int(bench[-1])
bench = bench[:-1]
chkpt = sys.argv[2] #0 indexed of last taken simpoint
offset = sys.argv[3]
spec_path = "/work/muke/spec2017/"
chkpt_path = "/work/muke/checkpoints/"+bench+"/"
gem5 = "/work/muke/PND-Loads/gem5-gen/"

os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
if type(specinvoke) != subprocess.CompletedProcess: Popen.wait(specinvoke)
commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
command = commands[run]
p = Popen(gem5+"build/ARM/gem5.fast --outdir="+chkpt_path+"checkpoints."+str(run)+".adjusted "+gem5+"configs/deprecated/example/se.py --cpu-type=NonCachingSimpleCPU --restore-with-cpu=NonCachingSimpleCPU -r 1 --simpoint-offset "+offset+" --last-simpoint "+chkpt+" --checkpoint-dir "+chkpt_path+"checkpoints."+str(run)+" --take-simpoint-checkpoint=/work/muke/simpoints/"+bench+"."+str(run)+".simpts.adjusted,/work/muke/simpoints/"+bench+"."+str(run)+".weights.adjusted,100000000,10000000 -c "+command.split()[0]+" --options=\""+' '.join(command.split()[1:])+"\" --mem-size=50GB", shell=True)
p.wait()

