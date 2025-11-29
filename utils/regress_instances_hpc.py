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
from lime.lime_regress import LimeTextExplainer
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import gc

parser = argparse.ArgumentParser(prog='regress_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)
parser.add_argument('--sample-method', type=str, required=True)
parser.add_argument('--num-samples', type=int, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
if args.branches:
    good_branches = args.branches.split(',')
elif args.branch_file:
    good_branches = [i.strip() for i in open(args.branch_file).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

sample_method = "slice"
if args.sample_method: sample_method = args.sample_method.split(',')[0]
if sample_method == "random": num_samples = 4000
elif sample_method == "slice": num_samples = 4000
else:
    print("Invalid sample method");
    exit(1)
if args.num_samples: num_samples = num_samples

workdir = os.getenv("PBS_O_WORKDIR")+"/"
tmpdir = os.getenv("TMPDIR")+"/"
confidence_dir = workdir+"/confidence-scores/"
perturbed_dir = workdir+"/perturbed-instances/"

dir_results = workdir+"/results/test/"+benchmark
dir_h5 = workdir+"/datasets/"+benchmark

sys.path.append(dir_results)
sys.path.append(os.getcwd())

from model import BranchNetTrainingPhaseKnobs

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

#parameters
threshold = logit(0.8)
num_features = config['history_lengths'][-1]

training_phase_knobs = BranchNetTrainingPhaseKnobs()

lime_explainer = LimeTextExplainer(
    class_names=["not_taken", "taken"],
    char_level=False,
    split_expression=lambda x: x.split(" "),
    bow=False,
    feature_selection="lasso_path",
    mask_string="0x000:not_taken",  # Mask string for unknown addresses
    sample_method=sample_method
)

def run_lime(row):

    exps = []

    instance = np.array(instances['full_history'])
    data = np.unpackbits(np.array(row['datas'], dtype=np.uint8)).reshape(num_samples, num_features)
    labels = np.array(row['perturbed_labels'], dtype=np.float32)
    zeros = np.zeros(labels.shape)
    labels = np.stack((zeros, labels), axis=1) #dumb but makes using lime easier

    exp = lime_explainer.explain_instance(instance,
                                          data, labels,
                                          num_features=num_features,
                                          num_samples=num_samples)

    return exp.as_list()

for branch in good_branches:

    print('Branch:', branch)

    input_file = perturbed_dir + "{}_branch_{}_{}_{}_perturbed_instances.parquet".format(benchmark, branch, run_type, sample_method)
    output_file = tmpdir+"{}_branch_{}_{}-{}_explained_instances.parquet".format(benchmark, branch, run_type, sample_method)
    chunk_size=5000

    first_explanation = pl.scan_parquet(input_file).limit(1).with_row_index("index").collect()
    sample = first_explanation.select(
        pl.struct(["index", "workload", "checkpoint", "label", "output", "history", "full_history", "weight", "datas", "perturbed_labels"]).map_elements(
            run_lime,
            return_dtype=pl.List(pl.Struct([pl.Field("feature",pl.Int64),pl.Field("impact",pl.Float32)]))
        ).alias("explanations")
    )   

    first_explanation = first_explanation.drop(first_explanation.columns[-2:])
    first_explanation = first_explanation.hstack(sample)
         
    schema = first_explanation.to_arrow().schema
    
    # Process in chunks and write to single file
    with pq.ParquetWriter(output_file, schema, compression='zstd') as writer:
        offset = 0
        while True:
            # Read chunk lazily
            chunk_lazy = pl.scan_parquet(input_file).slice(offset, chunk_size).with_row_index("index")
            
            try:
                chunk = chunk_lazy.collect()
                if chunk.height == 0:
                    break
            except:
                break

            # Process chunk
            processed = chunk.select(
                pl.struct(["index", "workload", "checkpoint", "label", "output", "history", "full_history", "weight", "datas", "perturbed_labels"]).map_elements(
                    run_lime,
                    return_dtype=pl.List(pl.Struct([pl.Field("feature",pl.Int64),pl.Field("impact",pl.Float32)]))
                ).alias("explanations")
            )   

            chunk = chunk.drop(chunk.columns[-2:])
            chunk = chunk.hstack(processed)

            # Write chunk to file immediately
            writer.write_table(chunk.to_arrow())
            
            offset += chunk_size
            
            gc.collect()

    subprocess.run("cp "+output_file+" "+workdir+"explanations/", shell=True, check=True)
