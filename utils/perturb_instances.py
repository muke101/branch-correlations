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

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--percentile', type=int, required=True)
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)
parser.add_argument('--sample-method', type=str, required=False)
parser.add_argument('--num-samples', type=int, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
device = str(args.device)
percentile = args.percentile
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
total_memory = torch.cuda.get_device_properties('cuda:'+device).total_memory
mem_per_instance = 0.6*1e6
batch_size = int(total_memory//mem_per_instance)
percentile = 100 - percentile

training_phase_knobs = BranchNetTrainingPhaseKnobs()
model = BranchNet(config, training_phase_knobs)
model.to('cuda:'+device)

lime_explainer = LimePerturber(
    class_names=["not_taken", "taken"],
    char_level=False,
    split_expression=lambda x: x.split(" "),
    bow=False,
    feature_selection="lasso_path",
    mask_string="0x000:not_taken",  # Mask string for unknown addresses
    sample_method=sample_method
)

def run_lime(instances, eval_wrapper, num_features, num_samples):

    histories = []
    interval = batch_size // num_samples
    writer = None
    schema = None
    file_name = "{}_branch_{}_{}-{}_explained_instances_top{}.parquet".format(benchmark, branch, run_type, sample_method, str(100 - percentile))
    
    try:
        for i, row in enumerate(instances.iter_rows()):
            history = np.array(row[-2], dtype=np.int64)
            histories.append(history)
            
            if len(histories) == interval:
                for data, perturbed_labels in lime_explainer.perturb_instances(
                    histories, eval_wrapper.probs_from_list_of_strings,
                    num_features=num_features, num_samples=num_samples,
                    batch_size=batch_size
                ):
                    # Convert to Arrow table
                    table = pa.table({
                        "datas": [np.packbits(row) for row in datas],
                        "perturbed_labels": perturbed_labels
                    })
                    
                    # Initialize writer with schema from first batch
                    if writer is None:
                        schema = table.schema
                        output_path = tmpdir+file_name
                        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                    
                    # Write batch
                    writer.write_table(table)
                
                histories = []  # Clear memory immediately
        
        # Handle remainder
        if len(histories) > 0:
                for data, perturbed_labels in lime_explainer.perturb_instances(
                    histories, eval_wrapper.probs_from_list_of_strings,
                    num_features=num_features, num_samples=num_samples,
                    batch_size=batch_size
                ):
                    table = pa.table({
                        "datas": [np.packbits(row) for row in datas],
                        "perturbed_labels": perturbed_labels
                    })
                    
                    writer.write_table(table)
            
    finally:
        if writer:
            writer.close()
            subprocess.run("cp "+tmpdir+file_name+" "+workdir+"perturbed_instances/", shell=True, check=True)

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    eval_wrapper = EvalWrapper.from_checkpoint(dir_ckpt, device, config_path=dir_config)

    # header: workload, checkpoint, label, output, history
    confidence_scores = pl.read_parquet(confidence_dir + "{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark, branch, run_type))

    #confidence_scores = confidence_scores.slice(0,100)

    print("Running lime")

    correlated_branches = run_lime(confidence_scores, eval_wrapper, num_features, num_samples)
