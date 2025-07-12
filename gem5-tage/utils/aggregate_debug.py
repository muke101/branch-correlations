import os
import collections
import sys
import re
from collections import Counter

h2ps = set()
h2p_indexes_file = "/home/bj321/Developer/branch-transformer-addon/h2p_indexes.txt"

with open(h2p_indexes_file, 'r') as file:
    for line in file.readlines():
        h2ps.add(hex(int(line.split(' ')[0])))

print(h2ps)

bench = sys.argv[1]
run = sys.argv[2]

aggregated_values = collections.defaultdict(float)

mode = "-pnd"
if "base" in os.getcwd(): mode = "-base"
mode = "-base"
bench_name = bench.split("_")[0].split(".")[1]
print("Benchmark: "+bench_name+"."+run+mode)

pattern_counts = Counter(
        {
            "pred:1, taken:0": 0,
            "pred:1, taken:1": 0,
            "pred:0, taken:0": 0,
            "pred:0, taken:1": 0,
        }
    )

results = {key: Counter(pattern_counts) for key in h2ps}

#pass over each .out and aggregate weighted values
for dirname in os.listdir("."):
    if os.path.isdir(dirname) and dirname[0].isdigit() and dirname.split('.')[1] == 'out':
        temp_results = {key: Counter(pattern_counts) for key in h2ps}
        debug_file = os.path.join(dirname, "debug.txt")
        chkpt_number = int(dirname.split('.')[0])
        if bench_name == "exchange2": cpt_dir = "/mnt/data/checkpoints-expanded/"+bench+"/checkpoints."+run
        else: cpt_dir = "/mnt/data/checkpoints/"+bench+"/checkpoints."+run
        for cpt in os.listdir(cpt_dir):
            if cpt.startswith("cpt.") and int(cpt.split('_')[1]) == (chkpt_number-1):
                weight = float(cpt.split('_')[5])
        print(debug_file)
        f = open(debug_file)
        # lines = f.readlines()
        # if len(lines) == 0:
        #     print("Checkpoint "+str(chkpt_number)+" has empty stats file!")
        #     exit(1)
        stats = {}
        count = 0
        for line in f:
            if "Commit branch" in line:
                for h2p in h2ps:
                    if h2p in line:
                        for pattern in pattern_counts:
                            if pattern in line:
                                temp_results[h2p][pattern] += 1
            #ignore empty lines and lines starting with "---"
            #regex would catch empty lines but we want to count number of begin/end markers for correctness checking
        #     if len(line.strip()) == 0: continue
        #     if not ignores.match(line):
        #         try:
        #             statKind = statLine.match(line).group(1)
        #             statValue = statLine.match(line).group(2)
        #         except AttributeError:
        #             continue
        #         #ignore atomic cpu stats
        #         if "system.cpu." in statKind: continue
        #         if statValue == 'nan':
        #             statValue = '0'
        #         stats[statKind] = float(statValue) * weight
        #     else:
        #         count += 1
        # if count != 4:
        #     print("Checkpoint "+str(chkpt_number)+" only has warmup!")
        #     exit(1)
        f.close()

        print("removing debug file: "+debug_file)
        os.remove(debug_file)

        for h2p in h2ps:
            for pattern in pattern_counts:
                results[h2p][pattern] += temp_results[h2p][pattern] * weight

        #collecting this passing over to clobber warmup values
        # for stat in stats:
        #     aggregated_values[stat] += stats[stat]

#handle CPI separately so we dont have to worry about averaging ratios
# aggregated_values["CPI"] = aggregated_values.get("system.switch_cpus.numCycles") / aggregated_values.get("system.switch_cpus.commitStats0.numInstsNotNOP")

# Write the results to the output file
results_file = open("results_debug.txt", "w")


for h2p in sorted(h2ps):
    for pattern in pattern_counts:
        results_file.write(f"{h2p} {pattern} {results[h2p][pattern]}\n")
    accuracy = (results[h2p]["pred:1, taken:1"] + results[h2p]["pred:0, taken:0"]) / (results[h2p]["pred:1, taken:1"] + results[h2p]["pred:1, taken:0"] + results[h2p]["pred:0, taken:0"] + results[h2p]["pred:0, taken:1"])
    results_file.write(f"{h2p} accuracy {accuracy}\n")
results_file.close()
# for field_name, value in aggregated_values.items():
#     results_file.write(f"{field_name} {value}\n")
# results_file.close()