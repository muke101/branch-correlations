import yaml
import torch
import get_traces
import os
import sys
import statistics
from collections import defaultdict
from scipy.stats import gmean
from scipy.special import logit
import numpy as np
import polars as pl
from lime_functions import EvalWrapper, dir_config, tensor_to_string
from lime.lime_eval import LimePerturber
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import multiprocessing as mp
import queue

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--ngpus', type=int, required=True)
parser.add_argument('--percentile', type=int, required=True)
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)
parser.add_argument('--sample-method', type=str, required=False)
parser.add_argument('--num-samples', type=int, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
percentile = args.percentile
ngpus = int(args.ngpus)
if args.branches:
    good_branches = args.branches.split(',')
elif args.branch_file:
    good_branches = [i.strip() for i in open(args.branch_file).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

sample_method = "slice"
if args.sample_method: sample_method = args.sample_method.split(',')[0]
if sample_method == "random": num_samples = 4000
elif sample_method == "slice": num_samples = 1000
else:
    print("Invalid sample method");
    exit(1)
if args.num_samples: num_samples = num_samples

workdir = os.getenv("PBS_O_WORKDIR")+"/"
tmpdir = os.getenv("TMPDIR")+"/"
confidence_dir = workdir+"/confidence-scores/"

dir_results = workdir+"/results/test/"+benchmark
dir_h5 = workdir+"/datasets/"+benchmark

sys.path.append(dir_results)
sys.path.append(os.getcwd())

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs
from benchmark_branch_loader import BenchmarkBranchLoader

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

#parameters 
threshold = logit(0.8)
num_features = config['history_lengths'][-1]
percentile = 100 - percentile

training_phase_knobs = BranchNetTrainingPhaseKnobs()

def writer(result_queue, output_path):

    writer = None

    try:
        while True:
            try:
                table = result_queue.get(timeout=30)  # Timeout to detect completion
                if table is None:  # Sentinel value
                    break
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)

            except queue.Empty:
                continue  # Keep waiting

    finally:
        if writer:
            writer.close()
            subprocess.run("cp "+output_path+" "+workdir+"perturbed-instances/", shell=True, check=True)

def run_lime(instances, branch, result_queue, device, num_features, num_samples):

    histories = []
    torch.cuda.set_device(device)
    lime_explainer = LimePerturber(
        class_names=["not_taken", "taken"],
        char_level=False,
        split_expression=lambda x: x.split(" "),
        bow=False,
        feature_selection="lasso_path",
        mask_string="0x000:not_taken",  # Mask string for unknown addresses
        sample_method=sample_method
    )
    dir_ckpt = dir_results + '/checkpoints/base_{}_checkpoint.pt'.format(branch)
    eval_wrapper = EvalWrapper.from_checkpoint(dir_ckpt, str(device), config_path=dir_config)
    total_memory = torch.cuda.get_device_properties('cuda:'+str(device)).total_memory
    mem_per_instance = 0.6*1e6 #inference size
    batch_size = int(total_memory//mem_per_instance)
    interval = batch_size // num_samples

    for i, row in enumerate(instances.iter_rows()):
        history = np.array(row[-2], dtype=np.int64)
        histories.append(history)

        if len(histories) == interval:
            for data, perturbed_labels in lime_explainer.perturb_instances(
                histories, eval_wrapper.probs_from_list_of_strings,
                num_features=num_features, num_samples=num_samples,
                batch_size=batch_size
            ):
                table = pa.table({
                    "datas": [np.packbits(data)],
                    "perturbed_labels": [perturbed_labels]
                })
                result_queue.put(table)

            histories = []

    # Handle remainder
    if len(histories) > 0:
            for data, perturbed_labels in lime_explainer.perturb_instances(
                histories, eval_wrapper.probs_from_list_of_strings,
                num_features=num_features, num_samples=num_samples,
                batch_size=batch_size
            ):
                table = pa.table({
                    "datas": [np.packbits(data)],
                    "perturbed_labels": [perturbed_labels]
                })
                result_queue.put(table)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    for branch in good_branches:

        print('Branch:', branch)

        # header: workload, checkpoint, label, output, history
        instances = pl.read_parquet(confidence_dir + "{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark, branch, run_type))

        #instances = instances.slice(0,100)

        slice_size = len(instances) // ngpus

        if ngpus > 1:
            result_queue = mp.Queue(maxsize=50)

            output_path = tmpdir+"/{}_branch_{}_{}-{}_explained_instances_top{}.parquet".format(benchmark, branch, run_type, sample_method, str(100 - percentile))

            writer_proc = mp.Process(target=writer,
                                     args=(result_queue, output_path))
            writer_proc.start()

            processes = []
            for device in range(ngpus):

                if device < ngpus-1:
                    instances_slice = instances.slice(device*slice_size, (device+1)*slice_size)
                else: #allocate remainder
                    instances_slice = instances.slice(device*slice_size, len(instances))

                    proc = mp.Process(target=run_lime,
                                      args=(instances_slice, branch, result_queue, device, num_features, num_samples))
                    proc.start()
                    processes.append(proc)

                    for proc in processes:
                        proc.join()

                        if ngpus > 1:
                            result_queue.put(None)
                            writer_proc.join()
