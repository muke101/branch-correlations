import subprocess
from subprocess import Popen
import sys
import os

bench = sys.argv[1]
run = int(bench[-1])
bench = bench[:-1]
spec_path = "/work/muke/spec2017/"
gem5 = "/work/muke/PND-Loads/gem5/"

os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
if type(specinvoke) != subprocess.CompletedProcess: Popen.wait(specinvoke)
commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
command = commands[run].split('>')[0]
p = Popen(gem5+"build/ARM/gem5.fast --outdir=checkpoints."+str(run)+" "+gem5+"configs/deprecated/example/se.py --cpu-type=NonCachingSimpleCPU --take-simpoint-checkpoint=/work/muke/simpoints/"+bench+"."+str(run)+".simpts,/work/muke/simpoints/"+bench+"."+str(run)+".weights,100000000,10000000 -c "+command.split()[0]+" --options=\""+' '.join(command.split()[1:])+"\" --mem-size=50GB", shell=True)
p.wait()
