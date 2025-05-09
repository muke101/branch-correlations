import yaml
import get_traces

yaml_dir = "/work/muke/Branch-Correlations/BranchNet/environment_setup/"
trace_dir = "/mnt/data/results/branch-project/traces/"
dataset_dir = "/mnt/data/results/branch-project/datasets/"

benchmark_dict = {}
partition_dict = {}

def write_benchmarks(benchmark):
    sets = ["test", "train", "validate"]
    input_sets = {set_type: [] for set_type in sets}

    for inp_set in sets:
        inputs = []
        workloads = get_traces.get_by_workload(benchmark, inp_set)
        for workload in workloads:
            input_dict = {"name": workload, "simpoints": []}
            for trace, weight in workloads[workload]:
                simpoint_number = trace.split('.')[-2]
                dataset = benchmark+"/"+'.'.join(trace.split('.')[:-1])+'.hdf5'
                input_dict["simpoints"].append({"id": int(simpoint_number)-1, "path": trace_dir+trace, "dataset": dataset_dir+dataset, "type": "pinball", "weight": weight})
           # inputs.append(input_dict)
            input_sets[inp_set].append(input_dict)

    benchmark_dict[benchmark] = {"inputs": input_sets}

for bench in get_traces.benchmarks:
    if bench == "648.exchange2_s":
        print("Warning: the alberta workloads for exchange2 still don't get serialised safely in yaml automatically")
    write_benchmarks(bench)

with open(yaml_dir+"benchmarks_transformer.yaml", "w") as f:
    yaml.dump(benchmark_dict, f, sort_keys=False)
