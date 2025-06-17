import yaml
import torch
import get_traces
import os
import sys
import statistics
from collections import defaultdict
from scipy.stats import gmean
import numpy as np

benchmark = sys.argv[1]

class AggregateStats:
    # This class aggregates statistics across multiple branches for a specific benchmark.
    def __init__(self):
        self.stats = {} 
        self.confidence_average = []
        self.confidence_stddev = []
        self.confidence_75th_percentile = []
        self.selected_confidence_average = []
        self.selected_confidence_stddev = []
        self.percent_selected_detrimental_impact_average = []
        self.instance_gini_coeff = []
        self.checkpoint_gini_coeff = []
        self.workload_gini_coeff = []
        self.benchmark_gini_coeff = []

    def add(self, stat):
        self.confidence_average.append(stat.confidence_average) 
        self.confidence_stddev.append(stat.confidence_stddev) 
        self.confidence_75th_percentile.append(stat.confidence_75th_percentile) 
        self.selected_confidence_average.append(stat.selected_confidence_average) 
        self.selected_confidence_stddev.append(stat.selected_confidence_stddev) 
        self.percent_selected_detrimental_impact_average.append(stat.percent_selected_detrimental_impact_average)
        self.instance_gini_coeff.append(stat.instance_gini_coeff)
        self.checkpoint_gini_coeff.append(stat.checkpoint_gini_coeff)
        self.workload_gini_coeff.append(stat.workload_gini_coeff)
        self.benchmark_gini_coeff.append(stat.benchmark_gini_coeff)
        self.stats[stat.pc] = stat

    def finalise(self):
        log_averages = np.log(np.array(self.confidence_average))
        log_stddevs = np.array(self.confidence_stddev) / np.array(self.confidence_average)
        log_mean_avg = np.mean(log_averages)
        log_stddev_avg = np.sqrt(np.mean(log_stddevs ** 2))
        self.confidence_average = np.exp(log_mean_avg)
        self.confidence_stddev = np.exp(log_stddev_avg)
        self.confidence_75th_percentile = gmean(self.confidence_75th_percentile)

        log_averages = np.log(np.array(self.selected_confidence_average))
        log_stddevs = np.array(self.selected_confidence_stddev) / np.array(self.selected_confidence_average)
        log_mean_avg = np.mean(log_averages)
        log_stddev_avg = np.sqrt(np.mean(log_stddevs ** 2))
        self.selected_confidence_average = np.exp(log_mean_avg)
        self.selected_confidence_stddev = np.exp(log_stddev_avg)
        self.instance_gini_coeff = gmean(self.instance_gini_coeff)
        self.checkpoint_gini_coeff = gmean(self.checkpoint_gini_coeff)
        self.workload_gini_coeff = gmean(self.workload_gini_coeff)
        self.benchmark_gini_coeff = gmean(self.benchmark_gini_coeff)

        self.percent_selected_detrimental_impact_average = gmean(self.percent_selected_detrimental_impact_average)

    def print(self):
        print("Average stats for benchmark "+benchmark+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\t75th Percentile of confidence: ", self.confidence_75th_percentile)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage percent of detrimental impact of selected features: ", self.percent_selected_detrimental_impact_average)
        print("\tAverage Gini coefficient per instance ", self.instance_gini_coeff)
        print("\tAverage Gini coefficient per checkpoint ", self.checkpoint_gini_coeff)
        print("\tAverage Gini coefficient per workload ", self.workload_gini_coeff)
        print("\tAverage Gini coefficient for benchmark ", self.benchmark_gini_coeff)
        print()

class Stats:
    # This class holds statistics for a specific branch.
    def __init__(self, pc):
        self.pc = pc
        self.confidence_average = []
        self.confidence_stddev = 0
        self.confidence_75th_percentile = 0
        self.selected_confidence_average = []
        self.selected_confidence_stddev = 0
        self.percent_selected_detrimental_impact_average = []
        self.instance_gini_coeff = []
        self.checkpoint_gini_coeff = []
        self.workload_gini_coeff = []
        self.benchmark_gini_coeff = 0

    def finalise(self):
        self.confidence_75th_percentile = np.quantile(self.confidence_average, 0.75)
        self.confidence_stddev = np.std(self.confidence_average)
        self.confidence_average = statistics.fmean(self.confidence_average)
        self.selected_confidence_stddev = np.std(self.selected_confidence_average)
        self.selected_confidence_average = statistics.fmean(self.selected_confidence_average)
        self.percent_selected_detrimental_impact_average = gmean(self.percent_selected_detrimental_impact_average)
        self.instance_gini_coeff = statistics.fmean(self.instance_gini_coeff)
        self.checkpoint_gini_coeff = statistics.fmean(self.checkpoint_gini_coeff)
        self.workload_gini_coeff = statistics.fmean(self.workload_gini_coeff)

    def print(self):
        print("Stats for branch "+self.pc+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\t75th Percentile of confidence: ", self.confidence_75th_percentile)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage percent of detrimental impact of selected features: ", self.percent_selected_detrimental_impact_average)
        print("\tAverage Gini coefficient per instance ", self.instance_gini_coeff)
        print("\tAverage Gini coefficient per checkpoint ", self.checkpoint_gini_coeff)
        print("\tAverage Gini coefficient per workload ", self.workload_gini_coeff)
        print("\tGini coefficient for benchmark ", self.benchmark_gini_coeff)
        print()

dir_results = '/mnt/data/results/branch-project/results/test/'+benchmark
dir_h5 = '/mnt/data/results/branch-project/datasets/'+benchmark
good_branches = ['0x41faa0'] #TODO: actually populate this somehow

sys.path.append(dir_results)
sys.path.append(os.getcwd())

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs
from benchmark_branch_loader import BenchmarkBranchLoader

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

training_phase_knobs = BranchNetTrainingPhaseKnobs()
model = BranchNet(config, training_phase_knobs)
model.to('cuda')

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

def filter_instances(loader, stats):
    num_instances = len(loader.instances)
    results = {}
    threshhold = 0.8 #FIXME: this is just a guess for now
    for c in range(0, num_instances):
        instance = loader.get_instance(c)
        history, label, workload, checkpoint = instance
        if workload not in results:
            results[workload] = {}
        if checkpoint not in results[workload]:
            results[workload][checkpoint] = [] #good instances
        with torch.no_grad():
            history = history.unsqueeze(0).cuda()
            if len(history.tolist()) != 1:
                print(history) 
                print("Found history with more dimensions than expected")
                exit(1)
            label = label.cuda()
            output = model(history)
            stats.confidence_average.append(output)
            if ((output > 0 and label == 1) or (output < 0 and label == 0)) and abs(output) > threshhold:
                results[workload][checkpoint].append(instance)
                stats.selected_confidence_average.append(output)
    return results

def coalecse_branches(correlated_branches, stats):

    # each checkpoint has many instances, with different selections of impactful branches.
    # take the average of absolute impactfulness in the right direction for each branch, to select the overall most impactful branch in that checkpoint
    # have to pass through the real results and penalize branches that go in the opposite direction

    for workload in correlated_branches:
        for checkpoint in correlated_branches[workload]:
            unique_branches = defaultdict(list)
            for instance in correlated_branches[workload][checkpoint]:
                impacts = []
                for pc, impact, label in instance:
                    correct_direction = (impact < 0 and label < 0) or (impact > 0 and label > 0)
                    impact = abs(impact)
                    impacts.append(impact)
                    if not correct_direction: impact *= -1
                    unique_branches[pc].append(impact)
                stats.instance_gini_coeff.append(gini(np.array(impacts)))

            for pc in unique_branches:
                unique_branches[pc] = statistics.fmean(unique_branches[pc])

            sorted_features = sorted(unique_branches.items(), key=lambda i: i[1], reverse=True)

            stats.checkpoint_gini_coeff.append(gini(np.array([i[1] for i in sorted_features])))

            correlated_branches[workload][checkpoint] = sorted_features

    return correlated_branches


def weight_branches(correlated_branches, stats):

    # collaspe per-checkpoint averages of instances into per-workload averages of checkpoints
    # for each checkpoint build a map of unique branches to checkpoint averages. then take the weighted gmean of this list by the simpoint weight.

    for workload in correlated_branches:
        unique_branches = defaultdict(lambda: ([],[])) #impact, weight
        for checkpoint in correlated_branches[workload]:
            weight = get_traces.get_simpoint_weight(benchmark, workload, checkpoint)
            for pc, impact in correlated_branches[workload][checkpoint]:
                unique_branches[pc][0].append(impact)
                unique_branches[pc][1].append(weight)
        for pc in unique_branches:
            unique_branches[pc] = np.exp(np.average(np.log(np.array(unique_branches[pc][0])), weights=np.array(unique_branches[pc][1])))

        correlated_branches[workload] = sorted(unique_branches.items(), key=lambda i: i[1], reverse=True)

        stats.workload_gini_coeff.append(gini(np.array([i[1] for i in correlated_branches[workload]])))

    return correlated_branches

def average_branches(correlated_branches, stats, use_train = True):
    # now we have a list of branches per workload, we can average them across workloads
    # this is the final step to get the most impactful branches for this H2P

    unique_branches = defaultdict(list)
    for workload in correlated_branches:
        if not use_train and 'train' not in workload: continue #eval workloads are actually called 'train'
        for pc, impact in correlated_branches[workload]:
            unique_branches[pc].append(impact)

    for pc in unique_branches:
        unique_branches[pc] = gmean(unique_branches[pc])

    sorted_features = sorted(unique_branches.items(), key=lambda i: i[1], reverse=True)

    gini_coeff = gini(np.array([i[1] for i in sorted_features]))

    stats.benchmark_gini_coeff = gini_coeff

    return (sorted_features, gini_coeff)

aggregate_stats = AggregateStats()

for branch in good_branches:

    print('Branch:', branch)

    stats = Stats(branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()
 
    train_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'train')
    eval_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'validate')

    # good_instaces -> {workload: {checkpoint: [instances]}}
    good_instances = filter_instances(train_loader, stats)
    good_instances.update(filter_instances(eval_loader, stats))

    # correlated_branches -> {workload: {checkpoint: [[num_feature most correlated branches] x num_instances]}}, this deepest dimension then has to get coalessed and then weighted
    correlated_branches = some_lime_function(good_instances, num_features = 50, samples = 10000)

    # combines results per-instances to select most impactful branches per checkpoint
    correlated_branches = coalecse_branches(correlated_branches, stats)

    correlated_branches = weight_branches(correlated_branches, stats)

    # finally, returns correlated_branches as a sorted list of branch pcs paired with impact
    correlated_branches, gini_coeff = average_branches(correlated_branches, stats)

    gamma = 0.3
    threshold = 1.0 - gamma * gini_coeff
    total = sum([i[1] for i in correlated_branches])
    cumulative = 0
    selected_branches = []
    for pc, impact in correlated_branches:
        cumulative += impact
        if cumulative / total < threshold:
            selected_branches.append((pc, impact))
        else:
            break

    print("Selected branches for branch {}:".format(branch), end=' ')
    c = 0
    for pc, impact in selected_branches:
        print("{}: {}".format(pc, impact), end=', ' if c < len(selected_branches) - 1 else '\n')
        c += 1

    stats.finalise()
    aggregate_stats.add(stats)

aggregate_stats.finalise()
aggregate_stats.print()
