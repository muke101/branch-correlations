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

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--percentile', type=int, required=True)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
device = str(args.device)
percentile = args.percentile

confidence_dir = "/mnt/data/results/branch-project/confidence-scores/"

dir_results = '/mnt/data/results/branch-project/results-x86/test/'+benchmark
dir_h5 = '/mnt/data/results/branch-project/datasets-x86/'+benchmark
#good_branches = ['0x41faa0'] #TODO: actually populate this somehow
good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

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
#num_samples = 4000
num_samples = 500
batch_size = 2**14
percentile = 100 - percentile

training_phase_knobs = BranchNetTrainingPhaseKnobs()
model = BranchNet(config, training_phase_knobs)
model.to('cuda:'+device)

lime_explainer = LimeTextExplainer(
    class_names=["not_taken", "taken"],
    char_level=False,
    split_expression=lambda x: x.split(" "),
    bow=False,
    feature_selection="lasso_path",
    mask_string="0x000:not_taken",  # Mask string for unknown addresses
)

def gini(array):
    array = array.flatten()
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def filter_instances(df):
    # remove rows where the confidence is below a threshold

    df = df.with_columns(pl.col('output').abs().alias('output'))

    percentile_value = np.percentile(np.array(df['output']), percentile)

    unfiltered_len = df.shape[0]
    filtered = df.filter(df['output'] > percentile_value)
    filtered_len = filtered.shape[0]

    print("Unfiltered instances: "+str(unfiltered_len))
    print("Filtered instances: "+str(filtered_len))
    print("Filtered " + str(unfiltered_len - filtered_len) + " instances")

    return filtered

def run_lime(instances, eval_wrapper, num_features, num_samples):

    exps = []
    interval = batch_size // num_samples
    unique_histories = {}
    histories = [np.array(instances['history'][0], dtype=np.int32)]

    for i in range(1, len(instances)):
        history = np.array(instances['history'][i], dtype=np.int32)
        hashable_history = tuple(history.tolist())
        if hashable_history in unique_histories: exps.append(unique_histories[hashable_history])
        else: histories.append(history)

        if len(histories) == interval:
            batch_exps = lime_explainer.explain_instances(histories,
                                        eval_wrapper.probs_from_list_of_strings,
                                        num_features=num_features, num_samples=num_samples)
            for c, exp in enumerate(batch_exps):
                exp = exp.as_list()
                unique_histories[tuple(histories[c].tolist())] = exp
                exps.append(exp)
            histories = []

    if len(histories) > 0: #clean up remainder
        exps.extend([exp.as_list() for exp in lime_explainer.explain_instances(histories, eval_wrapper.probs_from_list_of_strings, num_features=num_features, num_samples=num_samples)])

    return instances.with_columns(pl.Series("explanation", exps, dtype=pl.List(pl.Struct([pl.Field("feature",pl.Int16),pl.Field("impact",pl.Float64)]))))

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    eval_wrapper = EvalWrapper.from_checkpoint(dir_ckpt, config_path=dir_config, device=device)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()

    # header: workload, checkpoint, label, output, history
    confidence_scores = pl.read_parquet(confidence_dir + "{}_branch_{}_test_confidences_filtered.parquet".format(benchmark, branch))

    print("Filtering instances")
 
    confidence_scores = filter_instances(confidence_scores)

    print("Running lime")

    # correlated_branches -> {workload: {checkpoint: [[num_feature most correlated branches] x num_instances]}}, this deepest dimension then has to get coalessed and then weighted
    correlated_branches = run_lime(confidence_scores, eval_wrapper, num_features, num_samples)

    # Save the results
    correlated_branches.write_parquet("/mnt/data/results/branch-project/explained-instances/{}_branch_{}_{}_explained_instances_top{}.parquet".format(benchmark, branch, run_type, str(percentile)))
