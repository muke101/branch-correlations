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

hard_brs_dir = "/home/lp721/Branch-Correlations/h2ps/"

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmarks', type=str, required=False)
parser.add_argument('--ngpus', type=int, required=True)
#parser.add_argument('--sample-method', type=str, required=False)
parser.add_argument('--num-samples', type=int, required=False)

args = parser.parse_args()

if args.benchmarks:
    benchmarks = args.benchmarks.split(",")
else:
    benchmarks = get_traces.benchmarks
ngpus = int(args.ngpus)
num_samples = 4000
if args.num_samples: num_samples = num_samples
#sample_method = args.sample_method.split(',')[0]

workdir = "/mnt/datasets/lp721/"
confidence_dir = workdir+"/confidence-scores/"

if __name__ == "__main__":
    for benchmark in benchmarks:
        hard_branches = [i.strip() for i in open(hard_brs_dir+benchmark).readlines()[0].split(",")]
        for branch in hard_branches:
            for run_type in ["test"]:
                for sample_method in ["random", "slice"]:
                    subprocess.run("python3 /home/lp721/Branch-Correlations/utils/explain_instances_streamed.py --sample-method "+sample_method+" --benchmark "+benchmark+" --branches "+branch+" --run-type "+run_type+" --ngpus 4", shell=True, check=True)
                    output_path = workdir+"/explained-instances/{}_branch_{}_{}_{}_explained_instances.parquet".format(benchmark, branch, run_type, sample_method)
                    subprocess.run("rsync -e \"ssh -i /home/lp721/.ssh/doc\" --progress -av "+output_path+" muke@155.198.188.14:/mnt/data/results/branch-project/explained-instances/", shell=True, check=True)
                    subprocess.run("rm -f "+output_path, shell=True)
