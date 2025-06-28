import get_traces
import yaml
import torch
import os
import sys
from lime_functions import dir_config
import pickle
from dataset_loader import BranchDataset
import polars as pl
import numpy as np
from collections import defaultdict

torch.set_default_device('cuda')
batch_size = 2**14

benchmark = sys.argv[1]
dir_results = '/mnt/data/results/branch-project/results-x86/test/'+benchmark+"/"
confidence_dir = "/mnt/data/results/branch-project/confidence-scores/"
dir_h5 = '/mnt/data/results/branch-project/datasets-x86/'+benchmark+"/"
#good_branches = ['0x41faa0'] #TODO: actually populate this somehow
good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

sys.path.append(dir_results)
sys.path.append(os.getcwd())

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs

training_phase_knobs = BranchNetTrainingPhaseKnobs()
model = BranchNet(config, training_phase_knobs)
model.to('cuda')

def filter_instances(loader):

    history_indxs = {}
    history_list = []

    for batch_x, _, _, _ in loader:
        for i in range(len(batch_x)):
            history = tuple(batch_x[i].cpu().to_list())
            if history not in history_indxs:
                history_list.append(history)
                history_indxs[history] = len(history_list) - 1

    history_df = pl.DataFrame({
        "history": pl.Series("history", np.array(history_list))
    })

    unique_histories = {}
    workload_list = []
    checkpoint_list = []
    output_list = []
    label_list = []
    indx_list = []
    for batch_x, batch_y, checkpoints, workloads in loader:
        with torch.no_grad():
            outputs = model(batch_x)
        for i in range(len(outputs)):
            workload = workloads[i]
            if workload not in unique_histories:
                unique_histories[workload] = {}
            checkpoint = int(checkpoints[i])
            if checkpoint not in unique_histories[workload]:
                unique_histories[workload][checkpoint] = defaultdict(int)
            history = tuple(batch_x[i].cpu().to_list())
            output = outputs[i].cpu()
            label = batch_y[i].cpu()
            if ((output > 0 and label == 1) or (output < 0 and label == 0)):
                unique_histories[workload][checkpoint][history] += 1
                if unique_histories[workload][checkpoint][history] > 1: continue
                workload_list.append(workload)
                checkpoint_list.append(checkpoint)
                indx_list.append(history_indxs[history])
                output_list.append(float(output))
                label_list.append(int(label))

    weights = []
    for i in range(len(history_list)):
        workload = workload_list[i]
        checkpoint = checkpoint_list[i]
        history = history_list[i]
        total = sum(unique_histories[workload][checkpoint].values())
        weights.append(unique_histories[workload][checkpoint][history]/total)

    df = pl.DataFrame({
        "workload": np.array(workload_list),
        "checkpoint": np.array(checkpoint_list),
        "label": np.array(label_list),
        "output": np.array(output_list),
        "history_index": np.array(indx_list),
        "weight": np.array(weights)
    })

    return df

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()
 
    #train_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'train')
    #eval_loader = BenchmarkBranchLoader(benchmark, branch, dataset_type = 'validate')
    train_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'train')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    eval_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'validate')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    print("Num train instances: ", len(train_loader))
    print("Num eval instances: ", len(eval_loader))
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=batch_size, shuffle=False)

    print("Running train batches: ", len(train_loader))
    train_confidences = filter_instances(train_loader)
    del train_loader
    print("Running eval batches: ", len(eval_loader))
    eval_confidences = filter_instances(eval_loader)
    del eval_loader

    pl.concat([train_confidences, eval_confidences])

    train_confidences.write_parquet(confidence_dir+"{}_branch_{}_confidences_filtered.parquet".format(benchmark,branch))

    del train_confidences, eval_confidences
    torch.cuda.empty_cache()
