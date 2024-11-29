import subprocess
import os
import sys

addr_file_type = sys.argv[1]
cpu_model = sys.argv[2]
run_pnd = True
addr_file_dir = "/work/muke/PND-Loads/addr_files/"
chkpt_dir = "/work/muke/checkpoints/"
results_dir = "/work/muke/PND-Loads/results/"+addr_file_type+"/"+cpu_model+"/"
base_results_dir = "/work/muke/PND-Loads/results/base/"+cpu_model+"/"
benches = ["600.perlbench_s", "605.mcf_s", "619.lbm_s",
           "623.xalancbmk_s", "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s"] #"638.imagick_s", "644.nab_s"]

if addr_file_type == "base": exit(0) #nothing to compare to

prefix = "system.switch_cpus."

stats = {
    "CPI", prefix+"iew.memOrderViolationEvents",
    prefix+"MemDepUnit__0.MDPLookups", prefix+"executeStats0.numInsts",
}
stats_to_diff = {
    "CPI", prefix+"iew.memOrderViolationEvents",
    prefix+"MemDepUnit__0.MDPLookups",
}

def get_values(results):
    values = {}
    results = open(results, "r").readlines()
    for line in results:
        name = line.split()[0]
        value = line.split()[1]
        if name in stats:
            values[name] = float(value)
    return values

os.chdir(base_results_dir)
base_results = {}
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f):
        base_results[f] = get_values(f+"/results.txt")

os.chdir(results_dir)
differences = open("differences", "w")
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f):
        differences.write(f+":\n")
        base_result = base_results[f]
        pnd_result = get_values(f+"/results.txt")
        differences.write("\tBase CPI: "+str(base_result['CPI']+"\n"))
        differences.write("\tPND CPI: "+str(pnd_result['CPI']+"\n"))
        differences.write("\tBase Lookups Per KInst: "+str(base_result[prefix+'MemDepUnit__0.MDPLookups']/(base_result[prefix+'executeStats0.numInsts']*1000))+"\n")
        differences.write("\tPND Lookups Per KInst: "+str(pnd_result[prefix+'MemDepUnit__0.MDPLookups']/(pnd_result[prefix+'executeStats0.numInsts']*1000))+"\n")
        differences.write("\tBase Violations Per MInst: "+str(base_result[prefix+'iew.memOrderViolationEvents']/(base_result[prefix+'executeStats0.numInsts']*1000000))+"\n")
        differences.write("\tPND Violations Per MInst: "+str(pnd_result[prefix+'iew.memOrderViolationEvents']/(pnd_result[prefix+'executeStats0.numInsts']*1000000))+"\n")
        for field in pnd_result:
            if field not in stats_to_diff: continue
            base_value = base_result[field]
            pnd_value = pnd_result[field]
            difference = ((pnd_value - base_value) / base_value) * 100
            if "." in field: field = field.split(".")[-1]
            differences.write("\t"+field+": "+str(difference)+"\n")
        differences.write("\n")

differences.close()
