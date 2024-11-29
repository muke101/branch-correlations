import subprocess
import os
import sys
import argparse

parser =argparse.ArgumentParser(prog='cpu cpi cmp')
parser.add_argument('--ref-cpu', type=str, required=True)
parser.add_argument('--new-cpu', type=str, required=True)
parser.add_argument('--ref-run-type', type=str, required=True)
parser.add_argument('--new-run-type', type=str, required=True)
args = parser.parse_args()
ref_cpu = args.ref_cpu.split(',')[0]
new_cpu = args.new_cpu.split(',')[0]
ref_run_type = args.ref_run_type.split(',')[0]
new_run_type = args.new_run_type.split(',')[0]

addr_file_dir = "/work/muke/PND-Loads/addr_files/"
chkpt_dir = "/work/muke/checkpoints/"
results_dir = "/work/muke/results/"+new_run_type+"/base/"+new_cpu+"/"
results_dir_2 = "/work/muke/results/"+ref_run_type+"/base/"+ref_cpu+"/"
benches = ["600.perlbench_s", "605.mcf_s", "619.lbm_s",
           "623.xalancbmk_s", "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "644.nab_s"] #"638.imagick_s"]

prefix = "system.switch_cpus."

stats = {
    "CPI", #prefix+"iew.memOrderViolationEvents",
    prefix+"MemDepUnit__0.MDPLookups", prefix+"executeStats0.numInsts",
}
stats_to_diff = {
    "CPI", #prefix+"iew.memOrderViolationEvents",
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

os.chdir(results_dir_2)
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
        differences.write("\tBase CPI: "+str(base_result['CPI'])+"\n")
        differences.write("\tPND CPI: "+str(pnd_result['CPI'])+"\n")
        differences.write("\tBase Lookups Per KInst: "+str(base_result[prefix+'MemDepUnit__0.MDPLookups']/(base_result[prefix+'executeStats0.numInsts']*1000))+"\n")
        differences.write("\tPND Lookups Per KInst: "+str(pnd_result[prefix+'MemDepUnit__0.MDPLookups']/(pnd_result[prefix+'executeStats0.numInsts']*1000))+"\n")
        #differences.write("\tBase Violations Per MInst: "+str(base_result[prefix+'iew.memOrderViolationEvents']/(base_result[prefix+'executeStats0.numInsts']*1000000))+"\n")
        #differences.write("\tPND Violations Per MInst: "+str(pnd_result[prefix+'iew.memOrderViolationEvents']/(pnd_result[prefix+'executeStats0.numInsts']*1000000))+"\n")
        for field in pnd_result:
            if field not in stats_to_diff: continue
            base_value = base_result[field]
            pnd_value = pnd_result[field]
            try:
                difference = ((pnd_value - base_value) / base_value) * 100
            except:
                print(f)
                print(field)
                print(base_value)
                print(pnd_value)
            if "." in field: field = field.split(".")[-1]
            differences.write("\t"+field+": "+str(difference)+"\n")
        differences.write("\n")

differences.close()
