import os
import subprocess
from subprocess import Popen

spec = ["600.perlbench_s", "602.gcc_s", "605.mcf_s",
        "620.omnetpp_s", "623.xalancbmk_s", "625.x264_s",
        "631.deepsjeng_s", "641.leela_s", "657.xz_s"]
        #"648.exchange2_s"]
#remember, these paths are for the huawei server
spec_path = "/work/muke/spec2017-expanded/"
simpoint = "/work/muke/SimPoint/bin/simpoint"
workloads = "/work/muke/alberta-workloads/"
procs = []
bench_names = []

#def run_test():
#    sub_spec = ["602.gcc_s", "657.xs_s", "648.exchange2_s"]
#    for bench in sub_spec:
#        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/modified/run_peak_refspeed_mytest-64.0000")
#        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
#        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
#        for c, command in enumerate(commands):
#            command = command.split('>')[0]
#            bench_name = bench+"."+str(c)
#            bench_names.append(bench_name)
#            p = Popen(valgrind+" --tool=exp-bbv " + " --bb-out-file=/sim_home/luke/bbvs-expanded/bb.out."+bench_name+" "+command, shell=True)
#            procs.append(p)
#
#    return procs

def run_train():
    for bench in spec:
        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_train_mytest-64.0000")
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = (bench,"train."+str(c))
            bench_names.append(bench_name)

def run_alberta():
    for bench in spec:
        if bench == "600.perlbench_s": continue
        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
        stripped_name = bench.split('.')[1].split('_')[0]
        for workload in os.listdir(workloads+stripped_name):
            if bench == "602.gcc_s": binary = "sgcc_peak.mytest-64"
            else: binary = bench.split('.')[1]+"_peak.mytest-64"
            bench_name = (bench,workload)
            bench_names.append(bench_name)

run_train()
run_alberta()

for bench,workload in bench_names:
    subprocess.run("rm -f /work/muke/simpoints-expanded/"+bench+".simpts", shell=True)
    subprocess.run("rm -f /work/muke/simpoints-expanded/"+bench+".weights", shell=True)
    subprocess.run([simpoint, "-inputVectorsGzipped", "-loadFVFile",  "/mnt/data/checkpoints-expanded/"+bench+"/bbvs."+workload+"/simpoint.bb.gz",
            "-k", "search", "-maxK", "10", "-saveSimpoints",
            "/work/muke/simpoints-expanded/"+bench+"."+workload+".simpts", "-saveSimpointWeights",
            "/work/muke/simpoints-expanded/"+bench+"."+workload+".weights"])
