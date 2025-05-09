import yaml
import torch
import get_traces

dir_results = '/mnt/data/results/branch-project/results/test/648.exchange2_s'
dir_h5 = '/mnt/data/results/branch-project/datasets/648.exchange2_s'
#good_branches = ['0x429a78', '0x429b50']
good_branches = ['0x41faa0']

import os
import sys
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

for good_branch in good_branches:
    unique_histories = set()
    results = {} #per workload, per checkpoint
    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(good_branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()
 
    loader = BenchmarkBranchLoader('648.exchange2_s', good_branch)
    print('Branch:', good_branch)
    num_instances = len(loader.instances)
    for c in range(0, num_instances):
        history, label, workload, checkpoint = loader.get_instance(c)
        if workload not in results:
            results[workload] = {}
        if checkpoint not in results[workload]:
            results[workload][checkpoint] = [0,0]
        with torch.no_grad():
            history = history.unsqueeze(0).cuda()
            if len(history.tolist()) != 1:
                print(history) 
                exit(1)
            history_list = tuple(history[0].tolist())
            if history_list in unique_histories: continue
            unique_histories.add(history_list)
            label = label.cuda()
            output = model(history)
            results[workload][checkpoint][0] += 1
            if (output > 0 and label == 1) or (output < 0 and label == 0):
                results[workload][checkpoint][1] += 1

    total = 0
    correct = 0
    for workload in results:
        workload_total = 0
        workload_correct = 0
        for checkpoint in results[workload]:
            weight = get_traces.get_simpoint_weight('648.exchange2_s', workload, checkpoint) #TODO: parameterise benchmark
            results[workload][checkpoint][0] *= weight
            results[workload][checkpoint][1] *= weight
            workload_total += results[workload][checkpoint][0] 
            workload_correct += results[workload][checkpoint][1] 
        #FIXME: figure out if this is the correct way to aggregate across multiple workloads. probably isn't as weights should equal 1 within workloads. I think we're going off-script from branchnet with considering workloads together here.
        total += workload_total
        correct += workload_correct
    accuracy = (correct/total)*100
    print("Total accuracy for "+str(good_branch)+": ", accuracy)
    print("Total histories: ", len(loader.instances))
    print("Unique histories: ", len(unique_histories))
