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
confidence_dir = workdir+"confidence-scores/"

dir_results = workdir+"results/test/"+benchmark
dir_h5 = workdir+"datasets/"+benchmark
#good_branches = ['0x40a1ac'] #TODO: actually populate this somehow

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
num_samples = 4000
#num_samples = 500
batch_size = 2**14
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
)

def run_lime(instances, eval_wrapper, num_features, num_samples):

    all_perturbed_instances = []
    datas = []
    all_perturbed_labels = []
    interval = batch_size // num_samples
    histories = [np.array(instances['full_history'][0], dtype=np.int64)]

    for i in range(1, len(instances)):
        history = np.array(instances['full_history'][i], dtype=np.int64)
        histories.append(history)

        if len(histories) == interval:
            perturbed_instances, data, perturbed_labels = lime_explainer.perturb_instances(histories,
                                                                        eval_wrapper.probs_from_list_of_strings,
                                                                        num_features=num_features, num_samples=num_samples,
                                                                        batch_size=batch_size)
            all_perturbed_instances.append(perturbed_instances)
            datas.append(data)
            all_perturbed_labels.append(perturbed_labels)
            histories = []

    if len(histories) > 0: #clean up remainder
        perturbed_instances, data, perturbed_labels = lime_explainer.perturb_instances(histories,
                                                                    eval_wrapper.probs_from_list_of_strings,
                                                                    num_features=num_features, num_samples=num_samples,
                                                                    batch_size=batch_size)
        all_perturbed_instances.append(perturbed_instances)
        datas.append(data)
        all_perturbed_labels.append(perturbed_labels)

    return instances.hstack(pl.DataFrame({
        "perturbed_instances": all_perturbed_instances,
        "datas": datas,
        "perturbed_labels": all_perturbed_labels
    }))

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    eval_wrapper = EvalWrapper.from_checkpoint(dir_ckpt, device, config_path=dir_config)

    # header: workload, checkpoint, label, output, history
    confidence_scores = pl.read_parquet(confidence_dir + "{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark, branch, run_type))

    print("Running lime")

    # correlated_branches -> {workload: {checkpoint: [[num_feature most correlated branches] x num_instances]}}, this deepest dimension then has to get coalessed and then weighted
    correlated_branches = run_lime(confidence_scores, eval_wrapper, num_features, num_samples)

    # Save the results
    correlated_branches.write_parquet(workdir+"perturbed-instances/{}_branch_{}_{}_explained_instances_top{}.parquet".format(benchmark, branch, run_type, str(100 - percentile)))
