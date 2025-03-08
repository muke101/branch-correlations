import os
import re
import sys
from collections import defaultdict
trace_dir = "/mnt/data/results/branch-project/traces/"
hdf5_dir = "/mnt/data/results/branch-project/datasets/"
simpoint_dir = "/work/muke/simpoints-expanded/"
\
def get_simpoint_weight(benchmark, workload, checkpoint):
    simpt_file = open(simpoint_dir+benchmark+"."+workload+".simpts", "r")
    simpoints = [(int(s.strip().split()[0]), int(s.strip().split()[1])) for s in simpt_file.readlines()]
    simpoints = sorted(simpoints, key=lambda x: x[0])
    simpoint_indx = simpoints[int(checkpoint)-1][1]
    weight_file = open(simpoint_dir+benchmark+"."+workload+".weights", "r")
    weights = [w.split()[0] for w in weight_file.readlines()]
    simpt_file.close()
    weight_file.close()
    return float(weights[simpoint_indx])

def get_by_workload(benchmark, set_type):
    workload_dict = defaultdict(list)
    test_pattern = r"^\d$"

    for trace in os.listdir(trace_dir):
        if not trace.endswith('.trace') or not trace.startswith(benchmark): continue
        workload = trace.split('.')[2]
        if set_type == 'test':
            if not re.fullmatch(test_pattern, workload): continue
        elif set_type == 'validate':
            if benchmark == "600.perlbench_s":
                num = int(trace.split('.')[3])
                if workload == 'train' and num < 3:
                    workload = 'train.'+str(num)
                else: continue
            elif workload == 'train' and trace.split('.')[3].isdigit(): 
                workload = 'train.'+trace.split('.')[3]
            else: continue
        elif set_type == 'train':
            if benchmark == "600.perlbench_s": 
                num = int(trace.split('.')[3])
                if workload == 'train' and num >= 3:
                    workload = 'train.'+str(num)
                else: continue
            elif re.search(test_pattern, workload) or workload == 'train': continue
        else:
            print("Invalid set type!")
            exit(1)

        workload_dict[workload].append((trace, get_simpoint_weight(benchmark, workload, trace.split('.')[-2])))

    return workload_dict

def get_hdf5_set(benchmark, set_type):
    datasets = []
    test_pattern = r"^\d$"

    for trace in os.listdir(hdf5_dir):
        if not trace.endswith('.hdf5') or not trace.startswith(benchmark): continue
        workload = trace.split('.')[2]
        if set_type == 'test':
            if not re.fullmatch(test_pattern, workload): continue
        elif set_type == 'validate':
            if benchmark == "600.perlbench_s":
                num = int(trace.split('.')[3])
                if workload == 'train' and num < 3:
                    workload = 'train.'+str(num)
                else: continue
            elif workload == 'train' and trace.split('.')[3].isdigit(): 
                workload = 'train.'+trace.split('.')[3]
            else: continue
        elif set_type == 'train':
            if benchmark == "600.perlbench_s": 
                num = int(trace.split('.')[3])
                if workload == 'train' and num >= 3:
                    workload = 'train.'+str(num)
                else: continue
            elif re.search(test_pattern, workload) or workload == 'train': continue
        else:
            print("Invalid set type!")
            exit(1)
        datasets.append(trace)

    return datasets

def get_trace_set(benchmark, set_type):
    traces = []
    test_pattern = r"^\d$"

    for trace in os.listdir(trace_dir):
        if not trace.endswith('.trace') or not trace.startswith(benchmark): continue
        workload = trace.split('.')[2]
        if set_type == 'test':
            if not re.fullmatch(test_pattern, workload): continue
        elif set_type == 'validate':
            if benchmark == "600.perlbench_s":
                num = int(trace.split('.')[3])
                if workload == 'train' and num < 3:
                    workload = 'train.'+str(num)
                else: continue
            elif workload == 'train' and trace.split('.')[3].isdigit(): 
                workload = 'train.'+trace.split('.')[3]
            else: continue
        elif set_type == 'train':
            if benchmark == "600.perlbench_s": 
                num = int(trace.split('.')[3])
                if workload == 'train' and num >= 3:
                    workload = 'train.'+str(num)
                else: continue
            elif re.search(test_pattern, workload) or workload == 'train': continue
        else:
            print("Invalid set type!")
            exit(1)
        checkpoint = trace.split('.')[-2]
        weight = get_simpoint_weight(benchmark, workload, checkpoint)
        traces.append((trace, weight))

    return traces

if __name__ == "__main__":
    print(get_trace_set(sys.argv[1], sys.argv[2]))
    print(get_by_workload(sys.argv[1], sys.argv[2]))
