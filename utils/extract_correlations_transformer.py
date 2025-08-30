import yaml
import get_traces
import os
import sys
import statistics
from collections import defaultdict
import numpy as np
import polars as pl
from numba import jit
import argparse

selection_gamma = 2.5
threshold_percent = 0.7

parser = argparse.ArgumentParser(prog='extract_correlations', description='parse explained instances to find correlating branches for each H2P')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--percentile', type=int, required=True)
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
percentile = args.percentile
if args.branches:
    good_branches = args.branches.split(',')
elif args.branch_file:
    good_branches = [i.strip() for i in open(args.branch_file[0]).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

explain_dir = "/mnt/data/results/branch-project/explained-instances-indirect/"
dir_results = '/mnt/data/results/branch-project/results-indirect/test/'+benchmark
dir_h5 = '/mnt/data/results/branch-project/datasets-indirect/'+benchmark

sys.path.append(dir_results)
sys.path.append(os.getcwd())

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

#TODO: if we keep a Pattern class around for everything, need to create a workload -> checkpoint -> instances dictionary.
# alternatively, can create per-checkpoint classes and average those into new structures

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
        self.instance_gini_coeff = fast_mean(np.array(self.instance_gini_coeff))
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

@jit(nopython=True)
def fast_threshold_impacts(impacts, threshold_percent):
    """Optimized threshold_impacts using numba"""
    if len(impacts) == 0:
        return np.empty(0, dtype=np.int64)

    sorted_indices = np.argsort(impacts)[::-1]
    sorted_impacts = impacts[sorted_indices]

    # Fast cumulative sum
    cumulative = np.cumsum(sorted_impacts)
    total = cumulative[-1]
    threshold = threshold_percent * total

    # Find cutoff index
    cutoff_idx = 0
    for i in range(len(cumulative)):
        if cumulative[i] >= threshold:
            cutoff_idx = i
            break

    return sorted_indices[:cutoff_idx + 1]

@jit(nopython=True)
def fast_find_patterns(indices):
    """Optimized pattern finding that does all three pattern types at once"""
    n_indices = len(indices)

    # Initialize results
    offset = np.int64(-1)
    stride = np.int64(-1)
    group_start = np.int64(-1)
    group_end = np.int64(-1)

    if n_indices == 0:
        return offset, stride, group_start, group_end

    if n_indices == 1:
        offset = indices[0]
        return offset, stride, group_start, group_end

    # Check for stride pattern
    if n_indices >= 3:
        first_distance = indices[1] - indices[0]
        all_same = True
        for i in range(2, n_indices):
            if indices[i] - indices[i-1] != first_distance:
                all_same = False
                break
        if all_same:
            stride = first_distance

    # Check for group pattern
    start, end = indices[0], indices[-1]
    expected_size = end - start + 1
    if n_indices == expected_size:
        # Check if it's a contiguous range
        is_contiguous = True
        for i in range(n_indices - 1):
            if indices[i+1] - indices[i] != 1:
                is_contiguous = False
                break
        if is_contiguous:
            group_start = start
            group_end = end

    return offset, stride, group_start, group_end

class Pattern:
    def __init__(self, pc):
        self.pc = pc
        self.takenness = {'taken': [], 'not_taken': []}
        self.instance_takenness = {'taken': [], 'not_taken': []}
        self.checkpoint_takenness = {'taken': [], 'not_taken': []}
        self.workload_takenness = {'taken': [], 'not_taken': []}
        self.series = np.empty(582, dtype=np.float64)
        self.serise_indx = 0
        self.offsets = defaultdict(int)
        self.strides = defaultdict(int)
        self.groups = defaultdict(int)
        self.instance_lengths = []
        self.average_checkpoint_length = []

    def add(self, taken, impact):

        self.series[self.serise_indx] = impact
        self.serise_indx += 1

        if taken:
            self.takenness['taken'].append(impact)
        else:
            self.takenness['not_taken'].append(impact)

    def finalise_instance(self, weight):
        if len(self.takenness['taken']) > 0:
            taken_average = statistics.fmean(self.takenness['taken'])
            self.instance_takenness['taken'].append((taken_average, weight))
        if len(self.takenness['not_taken']) > 0:
            not_taken_average = statistics.fmean(self.takenness['not_taken'])
            self.instance_takenness['not_taken'].append((not_taken_average, weight))
        self.takenness = {'taken': [], 'not_taken': []}

        if self.serise_indx == 0:
            self.instance_lengths.append(0)
            return

        series = self.series[:self.serise_indx]
        thresholded_indices = fast_threshold_impacts(series, threshold_percent)
        offset, stride, group_start, group_end = fast_find_patterns(thresholded_indices)

        # Update pattern counters
        if offset != -1:
            self.offsets[offset] += weight
        else:
            self.offsets[-1] += weight

        if stride != -1:
            self.strides[stride] += weight
        else:
            self.strides[-1] += weight

        if group_start != -1 and group_end != -1:
            self.groups[(group_start, group_end)] += weight
        else:
            self.groups[-1] += weight

        self.instance_lengths.append(self.serise_indx)
        self.serise_indx = 0

    def finalise_checkpoint(self, weight):
        taken_impact = np.array([i[0] for i in self.instance_takenness['taken']])
        taken_weight = np.array([i[1] for i in self.instance_takenness['taken']])
        if len(taken_impact) > 0:
            self.checkpoint_takenness['taken'].append((fast_weighted_mean(taken_impact, weights=taken_weight), weight))
        not_taken_impact = np.array([i[0] for i in self.instance_takenness['not_taken']])
        not_taken_weight = np.array([i[1] for i in self.instance_takenness['not_taken']])
        if len(not_taken_impact) > 0:
            self.checkpoint_takenness['not_taken'].append((fast_weighted_mean(not_taken_impact, weights=not_taken_weight), weight))
        self.instance_takenness = {'taken': [], 'not_taken': []}
        self.average_checkpoint_length.append((statistics.fmean(self.average_instance_length), weight))
        self.average_instance_length = []

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
        #lengths = np.array([i[0] for i in self.average_checkpoint_length])
        #weights = np.array([i[1] for i in self.average_checkpoint_length])
        #self.average_checkpoint_length = np.average(lengths, weights=weights)

    def print(self):
        print("Patterns for branch "+hex(self.pc)+":")
        print("\tAverage series length: ", self.average_checkpoint_length)
        print("\tOffsets: ", self.offsets)
        print("\tStrides: ", self.strides)
        print("\tGroups: ", self.groups)
        print("\tTaken impact: ", statistics.fmean(self.workload_takenness['taken']))
        print("\tNot taken impact: ", statistics.fmean(self.workload_takenness['not_taken']))

    def print_instance(self):
        print("Patterns for branch "+hex(self.pc)+":")
        print("\tAverage series length: ", statistics.fmean(self.instance_lengths))
        print("\tOffsets: ", self.offsets)
        print("\tStrides: ", self.strides)
        print("\tGroups: ", self.groups)
        if len(self.instance_takenness['taken']) > 0:
            taken_impact = np.array([i[0] for i in self.instance_takenness['taken']])
            taken_weight = np.array([i[1] for i in self.instance_takenness['taken']])
            print("\tTaken impact: ", np.average(taken_impact, weights=taken_weight))
        if len(self.instance_takenness['not_taken']) > 0:
            not_taken_impact = np.array([i[0] for i in self.instance_takenness['not_taken']])
            not_taken_weight = np.array([i[1] for i in self.instance_takenness['not_taken']])
            print("\tNot taken impact: ", np.average(not_taken_impact, weights=not_taken_weight))


@jit(nopython=True)
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

@jit(nopython=True)
def fast_mean(array):
    return np.mean(array)

@jit(nopython=True)
def fast_weighted_mean(array, weights):
    return np.sum(array * weights) / np.sum(weights)

def coalecse_branches(explained_branches, patterns, stats):

    # each checkpoint has many instances, with different selections of impactful branches.
    # take the average of absolute impactfulness in the right direction for each branch, to select the overall most impactful branch in that checkpoint

    correlated_branches = {}

    workloads = explained_branches['workload'].unique()

    for workload in workloads:
        correlated_branches[workload] = {}

        # Filter once per workload
        workload_mask = explained_branches['workload'] == workload
        workload_instances = explained_branches.filter(workload_mask)

        checkpoints = workload_instances['checkpoint'].unique()

        for checkpoint in checkpoints:

            unique_branches = defaultdict(list)
            lengths = defaultdict(list)
            checkpoint_mask = workload_instances['checkpoint'] == checkpoint
            checkpoint_instances = workload_instances.filter(checkpoint_mask)
            rows = list(checkpoint_instances.iter_rows())

            for row in rows:
                # iterate over instance, collect all impacts per PC, record instance average along with instance weighting
                # then for each PC take the weighted gmean of average impacts across instances in the checkpoint
                label, history, impacts = row[2], row[5], row[6]

                #the most annoying unpacking in the world
                history = np.array([i[0] for i in history.to_list()[0]])
                impacts = np.array([i[0] for i in impacts.to_list()[0]])
                label = label[0][0]

                taken = history & 1
                pcs = history >> 1

                unique_pcs = np.unique(pcs)

                for pc in unique_pcs:
                    pc_mask = pcs == pc
                    series_length = int(pc_mask.sum())
                    pc_impacts_array = impacts[pc_mask]
                    pc_taken_array = taken[pc_mask]

                    avg_impact = fast_mean(pc_impacts_array)
                    unique_branches[pc].append(avg_impact)
                    lengths[pc].append(series_length)

                    #if pc not in patterns:
                    #    patterns[pc] = Pattern(pc)

                    #for taken_val, impact_val in zip(pc_taken_array, pc_impacts_array):
                    #    patterns[pc].add(taken_val, impact_val)

                    #patterns[pc].finalise_instance(weight)

                stats.instance_gini_coeff.append(gini(np.array(impacts)))

                if not unique_branches: continue

            for pc in unique_branches:
                #impacts_weights = np.array(unique_branches[pc])
                #avg_impacts = impacts_weights[:, 0]
                #weights = impacts_weights[:, 1]
                #unique_branches[pc] = (fast_weighted_mean(avg_impacts, weights=weights), fast_mean(np.array(lengths[pc]))) # weighted average
                avg_impacts = np.array(unique_branches[pc])
                unique_branches[pc] = (fast_mean(avg_impacts), fast_mean(np.array(lengths[pc]))) 
                lengths[pc] = fast_mean(np.array(lengths[pc]))

            sorted_features = sorted(unique_branches.items(), key=lambda i: i[1][0], reverse=True)

            impacts_array = np.array([impact for _, impact in sorted_features])
            stats.checkpoint_gini_coeff.append(gini(impacts_array))

            correlated_branches[workload][checkpoint] = sorted_features

    return correlated_branches

def weight_branches(correlated_branches, patterns, stats):

    # collaspe per-checkpoint averages of instances into per-workload averages of checkpoints
    # for each checkpoint build a map of unique branches to checkpoint averages. then take the weighted gmean of this list by the simpoint weight.

    for workload in correlated_branches:
        unique_branches = defaultdict(lambda: ([],[])) #impact, weight
        for checkpoint in correlated_branches[workload]:
            weight = get_traces.get_simpoint_weight(benchmark, workload, checkpoint)
            for pc, items in correlated_branches[workload][checkpoint]:
                unique_branches[pc][0].append(items)
                unique_branches[pc][1].append(weight)
                #patterns[pc].finalise_checkpoint(weight)
        for pc in unique_branches:
            #unique_branches[pc] = np.exp(np.average(np.log(np.array(unique_branches[pc][0])), weights=np.array(unique_branches[pc][1])))
            items = np.array(unique_branches[pc][0])
            weights = np.array(unique_branches[pc][1])
            impacts, lengths = np.split(items,2,axis=1)
            unique_branches[pc] = (fast_weighted_mean(impacts, weights), fast_weighted_mean(lengths, weights))
            #patterns[pc].finalise_workload()
            #if statistics.fmean(patterns[pc].instance_lengths) >= 5:
            #    patterns[pc].print_instance()

        correlated_branches[workload] = sorted(unique_branches.items(), key=lambda i: i[1][0], reverse=True)

        stats.workload_gini_coeff.append(gini(np.array([i[1][0] for i in correlated_branches[workload]])))

    return correlated_branches

def average_branches(correlated_branches, stats, use_train = True):
    # now we have a list of branches per workload, we can average them across workloads
    # this is the final step to get the most impactful branches for this H2P

    unique_branches = defaultdict(list)
    for workload in correlated_branches:
        if not use_train and 'train' not in workload: continue #eval workloads are actually called 'train'
        for pc, item in correlated_branches[workload]:
            unique_branches[pc].append(item)

    for pc in unique_branches:
        items = np.array(unique_branches[pc])
        impacts, lengths = np.split(items,2,axis=1)
        unique_branches[pc] = (fast_mean(impacts), fast_mean(lengths))

    sorted_features = sorted(unique_branches.items(), key=lambda i: i[1][0], reverse=True)

    gini_coeff = gini(np.array([i[1][0] for i in sorted_features]))

    stats.benchmark_gini_coeff = gini_coeff

    return (sorted_features, gini_coeff)

aggregate_stats = AggregateStats()

for branch in good_branches:

    print('Branch:', branch)

    stats = Stats(branch)

    # header: workload, checkpoint, label, output, history
    #explained_instances = pl.read_parquet(explain_dir + "{}_branch_{}_{}_explained_instances_top{}.parquet".format(benchmark, branch, run_type, str(percentile)))
    explained_instances = pl.read_parquet("/work/muke/Branch-Correlations/boyuan_transformer/0x417544/relevance.parquet")

    #stats.selected_confidence_average = explained_instances['output'].mean()
    #stats.selected_confidence_stddev = explained_instances['output'].std()

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
    total = sum([i[1][0] for i in correlated_branches])
    cumulative = 0
    selected_branches = []
    for pc, item in correlated_branches:
        #cumulative += impact
        #if cumulative / total < selection_threshold:
        selected_branches.append((pc, item))
        #else:
        #    break

    print("Selected branches for branch {}:".format(branch), end=' ')
    c = 0
    for pc, item in selected_branches:
        impact, length = item
        print("{}: {}".format(hex(pc), impact, length))#, end=', ' if c < len(selected_branches) - 1 else '\n')
        c += 1

    stats.finalise()
    stats.print()
    exit(1)
    aggregate_stats.add(stats)

aggregate_stats.finalise()
aggregate_stats.print()
