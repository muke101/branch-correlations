import yaml
import torch

dir_results = '/mnt/data/results/branch-project/results/test/648.exchange2_s'
dir_h5 = '/mnt/data/results/branch-project/datasets/648.exchange2_s'
good_branches = ['0x429a78', '0x429b50']

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
    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(good_branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()
 
    with BenchmarkBranchLoader('648.exchange2_s', good_branch) as loader:
        print('Branch:', good_branch)
        print('Instances:', len(loader))
        history, label = loader.get_instance(0)
        print('Example:', history, label)

        with torch.no_grad():
            history = history.unsqueeze(0).cuda()
            label = label.cuda()
            output = model(history)

            print('Model output:', output)

        

