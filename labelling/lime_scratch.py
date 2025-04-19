import yaml
import torch

dir_results = '/mnt/data/results/branch-project/results/test/648.exchange2_s'
good_branches = ['0x429a78', '0x429b50']

import sys
sys.path.append(dir_results)

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs
import h5py
import numpy as np

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

    # Path to your HDF5 file
    h5_path = '/path/to/your/data.h5'

    # Open HDF5 file
    with h5py.File(h5_path, 'r') as f:
        # Load data from HDF5 file
        # Adjust these keys based on your actual HDF5 structure
        inputs = torch.tensor(np.array(f['inputs']), dtype=torch.float32).to('cuda')
        labels = torch.tensor(np.array(f['labels']), dtype=torch.float32).to('cuda')
        
        # Print input shape
        print(f"Input shape: {inputs.shape}")
        
        # Run model inference
        with torch.no_grad():
            outputs = model(inputs)
        
        # Print results
        print(f"Output shape: {outputs.shape}")
        
        # You can calculate metrics
        # Example: if it's a classification task
        if outputs.dim() > 1 and outputs.size(1) > 1:
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            print(f'Accuracy: {correct / labels.size(0):.4f}')


