import pyyaml as yaml
import get_traces

yaml_dir = "/work/muke/Branch-Correlations/BranchNet/enviroment_setup/"
trace_dir = "/mnt/data/results/branch-project/traces/"

benchmark_dict = {}
partition_dict = {}

def write_benchmarks(benchmark):
    sets = ["test", "train", "validate"]
    inputs = []

    for inp_set in sets:
        workloads = get_traces.get_by_workload(benchmark, inp_set)
        for workload in workloads:
            input_dict = {"name": workload, "simpoints": []}
            for trace, weight in workloads[workload]:
                simpoint_number = trace.split('.')[-2]
                input_dict["simpoints"].append({"id": simpoint_number, "path": trace_dir+trace, "type": "pinball", "weight": weight})
            inputs.append(input_dict)

    benchmark_dict[benchmark] = {"inputs": inputs}

def write_partitions(benchmark):
    sets = ["test", "train", "validate"]

    partition_dict[benchmark] = {"spec_name": benchmark, "test_set": [], "train_set": [], "validation_set": []}

    for inp_set in sets:
        workloads = get_traces.get_by_workload(benchmark, inp_set)
        for workload in workloads:
            partition_dict[benchmark][inp_set+"_set"].append(workload)



for bench in get_traces.benchmarks:
    write_benchmarks(bench)
    write_partitions(bench)

with open(yaml_dir+"benchmarks.yaml", "w") as f:
    yaml.dump(benchmark_dict, f, sort_keys=False)

with open(yaml_dir+"ml_input_partitions.yaml", "w") as f:
    yaml.dump(partition_dict, f, sort_keys=False)
