import yaml
import torch
import os
import sys
from lime_functions import dir_config
import pickle

benchmark = sys.argv[1]
dir_results = '/mnt/data/results/branch-project/results-x86/test/'+benchmark
dir_h5 = '/mnt/data/results/branch-project/datasets-x86/'+benchmark
#good_branches = ['0x41faa0'] #TODO: actually populate this somehow
good_branches = [i for i in open(benchmark+"_branches").readlines()[0].split(",")]

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

def filter_instances(loader):

    num_instances = len(loader.instances)
    results = {}
    for c in range(0, num_instances):
        instance = loader.get_instance(c)
        history, label, workload, checkpoint = instance
        if workload not in results:
            results[workload] = {}
        if checkpoint not in results[workload]:
            results[workload][checkpoint] = [] 
        with torch.no_grad():
            history = history.unsqueeze(0).cuda()
            if len(history.tolist()) != 1:
                print(history) 
                print("Found history with more dimensions than expected")
                exit(1)
            label = label.cuda()
            output = model(history)
            if ((output > 0 and label == 1) or (output < 0 and label == 0)):
                results[workload][checkpoint].append((label,output))

    return results

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()
 
    train_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'train')
    eval_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'validation')

    confidences = filter_instances(train_loader)
    confidences.update(filter_instances(eval_loader))

    f = open("branch_{}_confidences.pickle".format(branch), "wb")
    pickle.dump(confidences, f)
    f.close()
