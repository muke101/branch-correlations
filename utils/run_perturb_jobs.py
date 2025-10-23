import get_traces
import argparse
import multiprocessing as mp
import queue
from lime_functions import EvalWrapper, dir_config, tensor_to_string
from lime.lime_eval import LimePerturber
import numpy as np
import polars as pl
import yaml
import torch
import get_traces
import os
import sys
import statistics
import subprocess

hard_brs_dir = "~/Branch-Correlations/h2ps/"

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--ngpus', type=int, required=True)
#parser.add_argument('--sample-method', type=str, required=False)
parser.add_argument('--num-samples', type=int, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
ngpus = int(args.ngpus)
num_samples = 4000
if args.num_samples: num_samples = num_samples
#sample_method = args.sample_method.split(',')[0]

workdir = "/mnt/dataset/lp721/"
confidence_dir = workdir+"/confidence-scores/"

def writer(result_queue):

    while True:
        try:
            path = result_queue.get(timeout=60*5)  # Timeout to detect completion
            if path is None:  # Sentinel value
                break
            subprocess.run("scp -i ~/.ssh/doc "+path+" muke@155.198.188.14:/mnt/data/results/branch-project/perturbed-instances/", shell=True, check=True)
            subprocess.run("rm -f "+path, shell=True)

        except queue.Empty:
            continue  # Keep waiting

if __name__ == "__main__":
    result_queue = mp.Queue(maxsize=50)
    writer_proc = mp.Process(target=writer, args=(result_queue))
    writer_proc.start()
    for benchmark in get_traces.benchmarks:
        hard_branches = [i.strip() for i in open(hard_brs_dir+benchmark).readlines()[0].split(",")]
        for branch in hard_branches:
            for run_type in ["test", "eval", "train"]:
                for sample_method in ["slice", "random"]:
                    subprocess.run("~/Branch-Correlations/utils/perturb_instances.py --sample-method "+sample_method+" --benchmark "+benchmark+" --branches "+branch+" --run-type "+run_type+" --ngpus 4", shell=True, check=True)
                    output_path = workdir+"/{}_branch_{}_{}_{}_perturbed_instances.parquet".format(benchmark, branch, run_type, sample_method)
                    result_queue.put(output_path)
    result_queue.put(None)
