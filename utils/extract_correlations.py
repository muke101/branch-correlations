import yaml
import get_traces
import os
import sys
import statistics
from collections import defaultdict
#from scipy.stats import gmean
#from scipy.special import logit
import numpy as np
import polars as pl
#from lime_functions import EvalWrapper, dir_config, tensor_to_string
#from lime.lime_text import LimeTextExplainer

selection_gamma = 2.5
threshold_percent = 0.7

benchmark = sys.argv[1]

explain_dir = "/mnt/data/results/branch-project/explained-instances/"

class AggregateStats:
    # This class aggregates statistics across multiple branches for a specific benchmark.
    def __init__(self):
        self.stats = {} 
        self.confidence_average = []
        self.confidence_stddev = []
        self.selected_confidence_average = []
        self.selected_confidence_stddev = []
        self.instance_gini_coeff = []
        self.checkpoint_gini_coeff = []
        self.workload_gini_coeff = []
        self.benchmark_gini_coeff = []

    def add(self, stat):
        self.confidence_average.append(stat.confidence_average) 
        self.confidence_stddev.append(stat.confidence_stddev) 
        self.selected_confidence_average.append(stat.selected_confidence_average) 
        self.selected_confidence_stddev.append(stat.selected_confidence_stddev) 
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


    def print(self):
        print("Average stats for benchmark "+benchmark+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage Gini coefficient per instance ", self.instance_gini_coeff)
        print("\tAverage Gini coefficient per checkpoint ", self.checkpoint_gini_coeff)
        print("\tAverage Gini coefficient per workload ", self.workload_gini_coeff)
        print("\tAverage Gini coefficient for benchmark ", self.benchmark_gini_coeff)
        print()

class Stats:
    # This class holds statistics for a specific branch.
    def __init__(self, pc):
        self.pc = pc
        self.confidence_average = 0
        self.confidence_stddev = 0
        self.selected_confidence_average = 0
        self.selected_confidence_stddev = 0
        self.instance_gini_coeff = []
        self.checkpoint_gini_coeff = []
        self.workload_gini_coeff = []
        self.benchmark_gini_coeff = 0

    def finalise(self):
        self.instance_gini_coeff = statistics.fmean(self.instance_gini_coeff)
        self.checkpoint_gini_coeff = statistics.fmean(self.checkpoint_gini_coeff)
        self.workload_gini_coeff = statistics.fmean(self.workload_gini_coeff)

    def print(self):
        print("Stats for branch "+self.pc+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage Gini coefficient per instance ", self.instance_gini_coeff)
        print("\tAverage Gini coefficient per checkpoint ", self.checkpoint_gini_coeff)
        print("\tAverage Gini coefficient per workload ", self.workload_gini_coeff)
        print("\tGini coefficient for benchmark ", self.benchmark_gini_coeff)
        print()

class Pattern:
    def __init__(self, pc):
        self.pc = pc
        self.takenness = {'taken': [], 'not_taken': []}
        self.instance_takenness = {'taken': [], 'not_taken': []}
        self.checkpoint_takenness = {'taken': [], 'not_taken': []}
        self.workload_takenness = {'taken': [], 'not_taken': []}
        self.series = []
        self.offsets = defaultdict(int)
        self.strides = defaultdict(int)
        self.groups = defaultdict(int)
        self.average_length = []

    def add(self, taken, impact):
        if taken:
            self.takenness['taken'].append(impact)
        else:
            self.takenness['not_taken'].append(impact)

        self.series.append(impact)

    def finalise_instance(self, weight):
        if len(self.takenness['taken']) > 0:
            self.instance_takenness['taken'].append((statistics.fmean(self.takenness['taken']), weight))
        if len(self.takenness['not_taken']) > 0:
            self.instance_takenness['not_taken'].append((statistics.fmean(self.takenness['not_taken']), weight))
        self.takenness = {'taken': [], 'not_taken': []}
        indecies = self.threshold_impacts()
        self.find_offset(indecies, weight)
        self.find_stride(indecies, weight)
        self.find_group(indecies, weight)
        self.average_length.append(len(self.series))
        self.series = []

    def finalise_checkpoint(self, weight):
        taken_impact = np.array([i[0] for i in self.instance_takenness['taken']])
        taken_weight = np.array([i[1] for i in self.instance_takenness['taken']])
        if len(taken_impact) > 0:
            self.checkpoint_takenness['taken'].append((np.average(taken_impact, weights=taken_weight), weight))
        not_taken_impact = np.array([i[0] for i in self.instance_takenness['not_taken']])
        not_taken_weight = np.array([i[1] for i in self.instance_takenness['not_taken']])
        if len(not_taken_impact) > 0:
            self.checkpoint_takenness['not_taken'].append((np.average(not_taken_impact, weights=not_taken_weight), weight))
        self.instance_takenness = {'taken': [], 'not_taken': []}

    def finalise_workload(self):
        taken_impact = np.array([i[0] for i in self.checkpoint_takenness['taken']])
        taken_weight = np.array([i[1] for i in self.checkpoint_takenness['taken']])
        if len(taken_impact) > 0:
            self.workload_takenness['taken'].append(np.average(taken_impact, weights=taken_weight))
        not_taken_impact = np.array([i[0] for i in self.checkpoint_takenness['not_taken']])
        not_taken_weight = np.array([i[1] for i in self.checkpoint_takenness['not_taken']])
        if len(not_taken_impact) > 0:
            self.workload_takenness['not_taken'].append(np.average(not_taken_impact, weights=not_taken_weight))
        self.checkpoint_takenness = {'taken': [], 'not_taken': []}

    def find_offset(self, indecies, weight):

        if len(indecies) == 1:
            self.offsets[indecies[0]] += weight
        else:
            self.offsets[-1] += weight

    def find_stride(self, indecies, weight):

        if len(indecies) == 1:
            self.strides[-1] += weight
            return

        stride = 0
        for i in range(len(indecies) - 2):
            distance = indecies[i+1] - indecies[i]
            if stride == 0:
                stride = distance
            elif stride != distance:
                self.strides[-1] += weight
                return
        self.strides[stride] += weight

    def find_group(self, indecies, weight):

        if len(indecies) == 1:
            self.groups[-1] += weight
            return

        start = indecies[0]
        end = indecies[-1]
        for i in range(start, end):
            if i not in indecies:
                self.groups[-1] += weight
                return

        self.groups[(start, end)] += weight

    def threshold_impacts(self):
    # This function returns the indecies of the impacts that are above a certain threshold.

        impacts = np.array(self.series)
        sorted_impacts = np.sort(impacts)[::-1]
        total = np.sum(sorted_impacts)
        cumulative = 0
        threshold = threshold_percent * total
        selected_indices = []
        for i in range(len(sorted_impacts)):
            cumulative += sorted_impacts[i]
            selected_indices.append(np.where(impacts == sorted_impacts[i])[0][0]) #assumes impacts are unique
            if cumulative >= threshold:
                break
        return selected_indices

    def print(self):
        print("Patterns for branch "+hex(self.pc)+":")
        #print("\tAverage series length: ", statistics.fmean(self.average_length))
        print("\tOffsets: ", self.offsets)
        print("\tStrides: ", self.strides)
        print("\tGroups: ", self.groups)
        print("\tTaken impact: ", statistics.fmean(self.workload_takenness['taken']))
        print("\tNot taken impact: ", statistics.fmean(self.workload_takenness['not_taken']))


dir_results = '/mnt/data/results/branch-project/results-x86/test/'+benchmark
dir_h5 = '/mnt/data/results/branch-project/datasets-x86/'+benchmark
#good_branches = ['0x41faa0'] #TODO: actually populate this somehow
good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

sys.path.append(dir_results)
sys.path.append(os.getcwd())

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

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

def coalecse_branches(explained_branches, patterns, stats):

    # each checkpoint has many instances, with different selections of impactful branches.
    # take the average of absolute impactfulness in the right direction for each branch, to select the overall most impactful branch in that checkpoint

    correlated_branches = {}

    for workload in explained_branches['workload'].unique():
        correlated_branches[workload] = {}
        workload_instances = explained_branches.filter(explained_branches['workload'] == workload)
        for checkpoint in workload_instances['checkpoint'].unique():
            correlated_branches[workload][checkpoint] = []
            checkpoint_instances = workload_instances.filter(workload_instances['checkpoint'] == checkpoint)
            unique_branches = defaultdict(list)
            for _, _, label, _, _, weight, explanation in checkpoint_instances.iter_rows():
                # iterate over instance, collect all impacts per PC, record instance average along with instance weighting
                # then for each PC take the weighted gmean of average impacts across instances in the checkpoint
                impacts = defaultdict(list)
                all_impacts = []
                for item in explanation:
                    feature = int(item['feature'])
                    impact = float(item['impact'])
                    taken = feature & 1
                    pc = feature >> 1
                    correct_direction = (impact < 0 and label == 0) or (impact > 0 and label == 1)
                    if not correct_direction: continue
                    impact = abs(impact)
                    impacts[pc].append(impact)
                    all_impacts.append(impact)
                    if pc not in patterns:
                        patterns[pc] = Pattern(pc)
                    patterns[pc].add(taken, impact)
                if len(impacts) == 0: continue #extreme corner case, shouldn't happen if we filter properly
                for pc in impacts:
                    unique_branches[pc].append((statistics.fmean(impacts[pc]), weight))
                    patterns[pc].finalise_instance(weight)
                stats.instance_gini_coeff.append(gini(np.array(all_impacts)))

            for pc in unique_branches:
                avg_impacts = np.array([i[0] for i in unique_branches[pc]])
                weights = np.array([i[1] for i in unique_branches[pc]])
                unique_branches[pc] = np.average(avg_impacts, weights=weights) # weighted average

            sorted_features = sorted(unique_branches.items(), key=lambda i: i[1], reverse=True)

            stats.checkpoint_gini_coeff.append(gini(np.array([i[1] for i in sorted_features])))

            correlated_branches[workload][checkpoint] = sorted_features

    return correlated_branches

def weight_branches(correlated_branches, patterns, stats):

    # collaspe per-checkpoint averages of instances into per-workload averages of checkpoints
    # for each checkpoint build a map of unique branches to checkpoint averages. then take the weighted gmean of this list by the simpoint weight.

    for workload in correlated_branches:
        unique_branches = defaultdict(lambda: ([],[])) #impact, weight
        for checkpoint in correlated_branches[workload]:
            weight = get_traces.get_simpoint_weight(benchmark, workload, checkpoint)
            for pc, impact in correlated_branches[workload][checkpoint]:
                unique_branches[pc][0].append(impact)
                unique_branches[pc][1].append(weight)
                patterns[pc].finalise_checkpoint(weight)
        for pc in unique_branches:
            #unique_branches[pc] = np.exp(np.average(np.log(np.array(unique_branches[pc][0])), weights=np.array(unique_branches[pc][1])))
            unique_branches[pc] = np.average(np.array(unique_branches[pc][0]), weights=np.array(unique_branches[pc][1]))
            patterns[pc].finalise_workload()
            if pc in [0x2bd, 0x591, 0x131]:
                patterns[pc].print()

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
        unique_branches[pc] = statistics.fmean(unique_branches[pc])

    sorted_features = sorted(unique_branches.items(), key=lambda i: i[1], reverse=True)

    gini_coeff = gini(np.array([i[1] for i in sorted_features]))

    stats.benchmark_gini_coeff = gini_coeff

    return (sorted_features, gini_coeff)

aggregate_stats = AggregateStats()

for branch in good_branches:

    print('Branch:', branch)

    stats = Stats(branch)

    # header: workload, checkpoint, label, output, history
    explained_instances = pl.read_parquet(explain_dir + "{}_branch_{}_test_explained_instances.parquet".format(benchmark, branch))

    stats.selected_confidence_average = explained_instances['output'].mean()
    stats.selected_confidence_stddev = explained_instances['output'].std()

    print("Averaging instances")

    patterns = {}

    # combines results per-instances to select most impactful branches per checkpoint
    correlated_branches = coalecse_branches(explained_instances, patterns, stats)

    del explained_instances

    print("Weighting checkpoints")

    correlated_branches = weight_branches(correlated_branches, patterns, stats)

    print("Averaging workloads")

    # finally, returns correlated_branches as a sorted list of branch pcs paired with impact
    correlated_branches, gini_coeff = average_branches(correlated_branches, stats)

    selection_threshold = 1.0 - selection_gamma * gini_coeff
    total = sum([i[1] for i in correlated_branches])
    cumulative = 0
    selected_branches = []
    for pc, impact in correlated_branches:
        cumulative += impact
        if cumulative / total < selection_threshold:
            selected_branches.append((pc, impact))
        else:
            break

    print("Selected branches for branch {}:".format(branch), end=' ')
    c = 0
    for pc, impact in selected_branches:
        print("{}: {}".format(hex(pc), impact), end=', ' if c < len(selected_branches) - 1 else '\n')
        c += 1

    stats.finalise()
    stats.print()
    exit(1)
    aggregate_stats.add(stats)

aggregate_stats.finalise()
aggregate_stats.print()
