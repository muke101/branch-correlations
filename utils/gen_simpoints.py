import os
import subprocess
from subprocess import Popen

"""
spec = ["600.perlbench_s", "602.gcc_s", "605.mcf_s",
        "619.lbm_s", "620.omnetpp_s",
        "623.xalancbmk_s", "625.x264_s", "631.deepsjeng_s",
        "638.imagick_s", "641.leela_s", "644.nab_s",
        "657.xz_s"]
"""
spec = ["602.gcc_s"]
spec_path = "/sim_home/luke/spec2017/"
valgrind = "/sim_home/luke/valgrind/build/bin/valgrind"
simpoint = "/sim_home/luke/SimPoint/bin/simpoint"
procs = []
bench_names = []

benches = {}
bbvs = {}
for bench in spec:
    os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
    specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
    Popen.wait(specinvoke)
    commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")] 
    for c, command in enumerate(commands):
        command = command.split('>')[0]
        bench_name = bench+"."+str(c)
        bench_names.append(bench_name)
        p = Popen(valgrind+" --tool=exp-bbv " + " --bb-out-file=/sim_home/luke/bbvs/bb.out."+bench_name+" "+command, shell=True)
        procs.append(p)

for p in procs:
    Popen.wait(p)

for bench in bench_names:
    Popen([simpoint, "-loadFVFile", "/sim_home/luke/bbvs/bb.out."+bench,
            "-k", "search", "-maxK", "30", "-saveSimpoints",
            "/sim_home/luke/simpoints/"+bench+".simpts", "-saveSimpointWeights",
            "/sim_home/luke/simpoints/"+bench+".weights"])
