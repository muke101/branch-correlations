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
from lime.lime_text import LimeTextExplainer
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
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)
parser.add_argument('--sample-method', type=str, required=False)
parser.add_argument('--num-samples', type=int, required=False)
 
args = parser.parse_args()
 
benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
ngpus = int(args.ngpus)
if args.branches:
    good_branches = args.branches.split(',')
elif args.branch_file:
    good_branches = [i.strip() for i in open(args.branch_file).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]
 
sample_method = "random"
if args.sample_method: sample_method = args.sample_method.split(',')[0]
if sample_method == "random": num_samples = 2000
elif sample_method == "slice": num_samples = 2000
else:
    print("Invalid sample method");
    exit(1)
if args.num_samples: num_samples = args.num_samples
 
workdir = "/mnt/datasets/lp721/"
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
num_features = config['history_lengths'][-1]
 
training_phase_knobs = BranchNetTrainingPhaseKnobs()
 
# Fix 2: mp.Queue initializer so pool workers share the queue without Manager overhead
_worker_queue = None
 
def pool_init(q):
    global _worker_queue
    _worker_queue = q
 
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
            #subprocess.run("cp "+output_path+" "+workdir+"perturbed-instances/", shell=True, check=True)
 
# Fix 4: accept parquet_path + slice indices instead of a pre-loaded DataFrame so
# the full dataset is never pickled and shipped to every worker process.
def run_lime(parquet_path, slice_start, slice_end, branch, device, num_features, num_samples):
 
    result_queue = _worker_queue  # Fix 2: retrieved from initializer global
 
    # Fix 4: each worker reads only its own slice from disk
    instances = pl.scan_parquet(parquet_path).slice(slice_start, slice_end - slice_start).collect()
 
    torch.cuda.set_device(device)
    lime_explainer = LimeTextExplainer(
        str(device),
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
 
    explanation_type = pa.list_(pa.struct([pa.field("feature", pa.int64()), pa.field("impact", pa.float64())]))
 
    for i in range(0, len(instances), interval):
        indxs = [i+j for j in range(interval) if i+j < len(instances)]
        histories = np.array([instances['full_history'][indx] for indx in indxs], dtype=np.int64)
        exps = lime_explainer.explain_instances(
                    histories, eval_wrapper.probs_from_list_of_strings,
                    num_features=num_features, num_samples=num_samples,
                    batch_size=batch_size)
 
        # Fix 1: accumulate all rows in this interval into a single table per put()
        # rather than one table per instance, drastically reducing queue pressure.
        rows = [instances.row(indx, named=True) for indx in indxs]
        input_cols = {k: pa.array([r[k] for r in rows]) for k in instances.columns}
        output_col = {"explanation": pa.array([exp.as_list() for exp in exps], type=explanation_type)}
        table = pa.table({**input_cols, **output_col})
        result_queue.put(table)
 
        del histories  # Fix 3: release array before the next allocation
 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
 
    for branch in good_branches:
 
        print('Branch:', branch)
        parquet_path = confidence_dir + "{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark, branch, run_type)
 
        # Read only the length so we can slice without shipping the full frame to workers
        total_rows = pl.scan_parquet(parquet_path).select(pl.first()).collect().height
 
        slice_size = total_rows // ngpus
        remainder = total_rows % ngpus
 
        # Fix 2: plain mp.Queue — no Manager process, no double-pickle overhead
        result_queue = mp.Queue(maxsize=50)
 
        output_path = workdir+"explained-instances/{}_branch_{}_{}_{}_explained_instances.parquet".format(benchmark, branch, run_type, sample_method)
        print("Writing to "+output_path)
 
        writer_proc = mp.Process(target=writer,
                                    args=(result_queue, output_path))
        writer_proc.start()
 
        worker_args = []
        start = 0
        for device in range(ngpus):
            end = start + slice_size + (1 if device < remainder else 0)
            # Fix 4: pass path + index range instead of a DataFrame slice
            worker_args.append((parquet_path, start, end, branch, device, num_features, num_samples))
            start = end
 
        try:
            # Fix 2: pass the queue via initializer so it isn't re-pickled per starmap call
            with mp.Pool(processes=ngpus, initializer=pool_init, initargs=(result_queue,)) as pool:
                pool.starmap(run_lime, worker_args)
        finally:
            result_queue.put(None)
            writer_proc.join()
            if writer_proc.exitcode != 0:
                print(f"Writer failed with exit code {writer_proc.exitcode}")
                exit(1)
