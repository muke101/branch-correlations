import os
import subprocess
from subprocess import Popen

spec = ["600.perlbench_s", "602.gcc_s", "605.mcf_s",
        "620.omnetpp_s", "623.xalancbmk_s", "625.x264_s",
        "631.deepsjeng_s", "641.leela_s", "657.xz_s"]
spec_path = "/work/muke/spec2017-x86/"
simpoint = "/work/muke/SimPoint/bin/simpoint"
simpoint_dir = "/work/muke/simpoints-x86"
procs = []
bench_names = []

def run_test():
    for bench in spec:
        run_dir = spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000"
        os.chdir(run_dir)
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = (bench,str(c))
            bench_names.append(bench_name)
run_test()

for bench,workload in bench_names:
    checkpoint_dir = "/mnt/data/checkpoints-expanded-x86/"
    subprocess.run([simpoint, "-loadFVFile",  checkpoint_dir+bench+"/bbvs."+workload,
            "-k", "search", "-maxK", "10", "-saveSimpoints",
            simpoint_dir+bench+"."+workload+".simpts", "-saveSimpointWeights",
            simpoint_dir+bench+"."+workload+".weights"], check=True)
