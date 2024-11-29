import subprocess
import os
import sys

run_type = sys.argv[1]
addr_file_type = sys.argv[2]
cpu_model = sys.argv[3]
results_dir = "/work/muke/results/"+run_type+"/"+addr_file_type+"/"+cpu_model+"/"
base_results_dir = "/work/muke/results/"+run_type+"/base/"+cpu_model+"/"
benches = ["600.perlbench_s", "605.mcf_s", "619.lbm_s",
           "623.xalancbmk_s", "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "644.nab_s"] #"638.imagick_s"]

#aggregate stats
for bench in benches:
    name = bench.split(".")[1].split("_")[0]

    for i in (0,1,2):
        if os.path.exists(results_dir+name+'.'+str(i)):
            raw_results_dir = results_dir+name+'.'+str(i)+"/raw/"
            os.chdir(raw_results_dir)
            p = subprocess.Popen("python3 /work/muke/PND-Loads/utils/aggregate_stats.py "+bench+" "+str(i), shell=True)
            p.wait()
            subprocess.Popen("cp results.txt ../", shell=True)
            raw_results_dir = base_results_dir+name+'.'+str(i)+"/raw/"
            os.chdir(raw_results_dir)
            p = subprocess.Popen("python3 /work/muke/PND-Loads/utils/aggregate_stats.py "+bench+" "+str(i), shell=True)
            p.wait()
            subprocess.Popen("cp results.txt ../", shell=True)

if addr_file_type == "base": exit(0) #nothing to compare to

prefix = "system.switch_cpus."

stats = {
    "CPI", prefix+"iew.memOrderViolationEvents",
    prefix+"MemDepUnit__0.MDPLookups", prefix+"executeStats0.numInsts",
    prefix+"MemDepUnit__0.SSITCollisions"
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
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        base_results[f] = get_values(f+"/results.txt")

os.chdir(results_dir)
differences = open("differences", "w")
for f in os.listdir(os.getcwd()):
    if os.path.isdir(f) and os.path.exists(f+"/results.txt"):
        differences.write(f+":\n")
        base_result = base_results[f]
        pnd_result = get_values(f+"/results.txt")
        differences.write("\tBase CPI: "+str(base_result['CPI'])+"\n")
        differences.write("\tPND CPI: "+str(pnd_result['CPI'])+"\n")
        differences.write("\tBase Lookups Per KInst: "+str(1024*base_result[prefix+'MemDepUnit__0.MDPLookups']/base_result[prefix+'executeStats0.numInsts'])+"\n")
        differences.write("\tPND Lookups Per KInst: "+str(1024*pnd_result[prefix+'MemDepUnit__0.MDPLookups']/pnd_result[prefix+'executeStats0.numInsts'])+"\n")
        differences.write("\tBase Violations Per MInst: "+str(1024*1024*base_result[prefix+'iew.memOrderViolationEvents']/base_result[prefix+'executeStats0.numInsts'])+"\n")
        differences.write("\tPND Violations Per MInst: "+str(1024*1024*pnd_result[prefix+'iew.memOrderViolationEvents']/pnd_result[prefix+'executeStats0.numInsts'])+"\n")
        differences.write("\tBase Collisions Per KInst: "+str(1024*base_result[prefix+'MemDepUnit__0.SSITCollisions']/base_result[prefix+'executeStats0.numInsts'])+"\n")
        differences.write("\tPND Collisions Per KInst: "+str(1024*pnd_result[prefix+'MemDepUnit__0.SSITCollisions']/pnd_result[prefix+'executeStats0.numInsts'])+"\n")
        for field in pnd_result:
            if field not in stats_to_diff: continue
            base_value = base_result[field]
            pnd_value = pnd_result[field]
            difference = ((pnd_value - base_value) / base_value) * 100
            if "." in field: field = field.split(".")[-1]
            differences.write("\t"+field+": "+str(difference)+"\n")
        differences.write("\n")

differences.close()
