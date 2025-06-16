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
    def __init__(self):
        self.stats = {} 
        self.confidence_average = []
        self.confidence_stddev = []
        self.confidence_75th_percentile = []
        self.selected_confidence_average = []
        self.selected_confidence_stddev = []
        self.percent_selected_detrimental_impact_average = []
        self.top80_instance_average_count = []
        self.top80_checkpoint_average_count = []
        self.top80_workload_average_count = []

    def add(self, stat):
        self.confidence_average.append(stat.confidence_average) 
        self.confidence_stddev.append(stat.confidence_stddev) 
        self.confidence_75th_percentile.append(stat.confidence_75th_percentile) 
        self.selected_confidence_average.append(stat.selected_confidence_average) 
        self.selected_confidence_stddev.append(stat.selected_confidence_stddev) 
        self.percent_selected_detrimental_impact_average.append(stat.percent_selected_detrimental_impact_average) 
        self.top80_instance_average_count.append(stat.top80_instance_average_count) 
        self.top80_checkpoint_average_count.append(stat.top80_checkpoint_average_count) 
        self.top80_workload_average_count.append(stat.top80_workload_average_count) 
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

        self.percent_selected_detrimental_impact_average = gmean(self.percent_selected_detrimental_impact_average)
        self.top80_instance_average_count = gmean(self.top80_instance_average_count)
        self.top80_checkpoint_average_count = gmean(self.top80_checkpoint_average_count)
        self.top80_workload_average_count = gmean(self.top80_workload_average_count)

    def print(self):
        print("Average stats for benchmark "+benchmark+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\t75th Percentile of confidence: ", self.confidence_75th_percentile)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage percent of detrimental impact of selected features: ", self.percent_selected_detrimental_impact_average)
        print("\tAverage per instance number of features capturing 80% of impact: ", self.top80_instance_average_count)
        print("\tAverage per checkpoint number of features capturing 80% of impact: ", self.top80_checkpoint_average_count)
        print("\tAverage per workload number of features capturing 80% of impact: ", self.top80_workload_average_count)
        print()

class Stats:
    def __init__(self, pc):
        self.pc = pc
        self.confidence_average = 0
        self.confidence_stddev = 0
        self.confidence_75th_percentile = 0
        self.selected_confidence_average = 0
        self.selected_confidence_stddev = 0
        self.percent_selected_detrimental_impact_average = 0
        self.top80_instance_average_count = 0
        self.top80_checkpoint_average_count = 0
        self.top80_workload_average_count = 0

    def finalise(self):
        self.confidence_75th_percentile = np.quantile(self.confidence_average, 0.75)
        self.confidence_stddev = np.std(self.confidence_average)
        self.confidence_average = statistics.fmean(self.confidence_average)
        self.selected_confidence_stddev = np.std(self.selected_confidence_average)
        self.selected_confidence_average = statistics.fmean(self.selected_confidence_average)
        self.percent_selected_detrimental_impact_average = gmean(self.percent_selected_detrimental_impact_average)
        self.top80_instance_average_count = statistics.fmean(self.top80_instance_average_count)
        self.top80_checkpoint_average_count = statistics.fmean(self.top80_checkpoint_average_count)
        self.top80_workload_average_count = statistics.fmean(self.top80_workload_average_count)

    def print(self):
        print("Stats for branch "+self.pc+":")
        print("\tAverage confidence: ", self.confidence_average)
        print("\tConfidence stddev: ", self.confidence_stddev)
        print("\t75th Percentile of confidence: ", self.confidence_75th_percentile)
        print("\tAverage confidence of selected instances: ", self.selected_confidence_average)
        print("\tStddev of selected instance confidence: ", self.selected_confidence_stddev)
        print("\tAverage percent of detrimental impact of selected features: ", self.percent_selected_detrimental_impact_average)
        print("\tAverage per instance number of features capturing 80% of impact: ", self.top80_instance_average_count)
        print("\tAverage per checkpoint number of features capturing 80% of impact: ", self.top80_checkpoint_average_count)
        print("\tAverage per workload number of features capturing 80% of impact: ", self.top80_workload_average_count)
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
            history_list = tuple(history[0].tolist())
            label = label.cuda()
            output = model(history)
            stats.confidence_average += output
            if ((output > 0 and label == 1) or (output < 0 and label == 0)) and abs(output) > threshhold:
                results[workload][checkpoint].append(instance)
                stats.selected_confidence_average += output
    return results

def coalecse_branches(correlated_branches):

    # each checkpoint has many instances, with different selections of impactful branches.
    # take the average of absolute impactfulness in the right direction for each branch, to select the overall most impactful branch in that checkpoint
    # have to pass through the real results and penalize branches that go in the opposite direction

    #TODO: figure out what impact represents. if its already a percentage of 100 then that makes things easy, otherwise have to calculate that myself to find top 80%
    
    for workload in correlated_branches:
        for checkpoint in correlated_branches[workload]:
            unique_branches = defaultdict(list)
            for instance in correlated_branches[workload][checkpoint]:
                #per instance goes here
                total_impact = 0
                i = 0
                for pc, impact in instance: 
                    unique_branches[pc].append(impact) 
                    total_impact += impact
                    i += 1 

            for pc in unique_branches:
                #per checkpoint goes here
                unique_branches[pc] = statistics.fmean(unique_branches[pc])
        #per workload goes here


def weight_branches(correlated_branches):
    
    for workload in correlated_branches:
        for checkpoint in correlated_branches[workload]:
            for instance in correlated_branches[workload][checkpoint]:
                pass
            

aggregated_stats = AggregatedStats()

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
    correlated_branches = some_lime_function(good_instances, num_features = 5)

    # combines results per-instances to select most impactful branches per checkpoint
    correlated_branches = coalesce_branches(correlated_branches)

    correlated_branches = weight_branches(correlated_branches)

    total = 0
    correct = 0
    # train.0, alberta.0...
    for workload in results:
        workload_total = 0
        workload_correct = 0
        for checkpoint in results[workload]:
            weight = get_traces.get_simpoint_weight(benchmark, workload, checkpoint) #TODO: parameterise benchmark
            results[workload][checkpoint][0] *= weight
            results[workload][checkpoint][1] *= weight
            workload_total += results[workload][checkpoint][0] 
            workload_correct += results[workload][checkpoint][1] 
        #FIXME: figure out if this is the correct way to aggregate across multiple workloads. probably isn't as weights should equal 1 within workloads. I think we're going off-script from branchnet with considering workloads together here.
        total += workload_total
        correct += workload_correct
    accuracy = (correct/total)*100
    print("Total accuracy for "+str(branch)+": ", accuracy)
    print("Total histories: ", len(loader.instances))
    print("Unique histories: ", len(unique_histories))

    stats.finalise()
    aggregated_stats.add(stats)

aggregated_stats.finalise()
aggregated_stats.print()
