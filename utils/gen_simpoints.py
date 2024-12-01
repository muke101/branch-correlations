import os
import subprocess
from subprocess import Popen

spec = ["600.perlbench_s", "602.gcc_s", "605.mcf_s",
        "620.omnetpp_s", "623.xalancbmk_s", "625.x264_s",
        "631.deepsjeng_s", "641.leela_s", "657.xz_s",
        "648.exchange2_s"]
#remember, these paths are for the huawei server
spec_path = "/sim_home/luke/spec2017/"
expanded_spec_path = "/sim_home/luke/spec2017-expanded/"
valgrind = "/sim_home/luke/valgrind/build/bin/valgrind"
simpoint = "/sim_home/luke/SimPoint/bin/simpoint"
workloads = "/sim_home/luke/alberta-workloads/"
procs = []
bench_names = []

def run_test():
    sub_spec = ["602.gcc_s", "657.xs_s", "648.exchange2_s"]
    for bench in sub_spec:
        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/modified/run_peak_refspeed_mytest-64.0000")
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+"."+str(c)
            bench_names.append(bench_name)
            p = Popen(valgrind+" --tool=exp-bbv " + " --bb-out-file=/sim_home/luke/bbvs-expanded/bb.out."+bench_name+" "+command, shell=True)
            procs.append(p)

    return procs

def run_train():
    for bench in spec:
        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_train_mytest-64.0000")
        specinvoke = subprocess.run([spec_path+"bin/specinvoke", "-n"], stdout=subprocess.PIPE)
        commands = [line.decode().strip() for line in specinvoke.stdout.split(b"\n") if line.startswith(b".")]
        for c, command in enumerate(commands):
            command = command.split('>')[0]
            bench_name = bench+".train."+str(c)
            bench_names.append(bench_name)
            p = Popen(valgrind+" --tool=exp-bbv " + " --bb-out-file=/sim_home/luke/bbvs-expanded/bb.out."+bench_name+" "+command, shell=True)
            procs.append(p)

    return procs

def run_alberta():
    for bench in spec:
        os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
        bench_name = bench.split('.')[1].split('_')[0]
        for workload in os.listdir(workloads+bench_name):
            subprocess.run("cp -r "+workloads+bench_name+"/"+workload+"/input/* .", shell=True)
            if bench == "602.gcc_s": binary = "sgcc_peak.mytest-64"
            else: binary = bench.split('.')[1]+"_s_peak.mytest-64"
            control = open("control", "r")
            flags = control.readlines()[0].strip()
            command = binary + ' ' + flags
            bench_name = bench+"."+workload
            bench_names.append(bench_name)
            if bench == "625.x264_s":
                input_name = workload[0].upper()+workload[1:]+".264"
                subprocess.run("./ldecod_s_peak.mytest-64 -i "+input_name+".264 -o "+input_name+".yuv", capture_output=True, shell=True)
            p = Popen(valgrind+" --tool=exp-bbv " + " --bb-out-file=/sim_home/luke/bbvs-expanded/bb.out."+bench_name+" "+command, shell=True)
            procs.append(p)

    return procs

procs += run_test()
procs += run_train()
procs += run_alberta()
for p in procs:
    Popen.wait(p)

for bench in bench_names:
    subprocess.run([simpoint, "-loadFVFile", "/sim_home/luke/bbvs-expanded/bb.out."+bench,
            "-k", "search", "-maxK", "10", "-saveSimpoints",
            "/sim_home/luke/simpoints-expanded/"+bench+".simpts", "-saveSimpointWeights",
            "/sim_home/luke/simpoints-expanded/"+bench+".weights"])
