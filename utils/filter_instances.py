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
import argparse

parser = argparse.ArgumentParser(prog='explain_instances', description='run lime forever and ever')

parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--branches', type=str, required=False)
parser.add_argument('--branch-file', type=str, required=False)

args = parser.parse_args()

benchmark = args.benchmark.split(',')[0]
run_type = args.run_type.split(',')[0]
device = str(args.device)
if args.branches:
    good_branches = args.branches.split(',')
elif args.branch_file:
    good_branches = [i.strip() for i in open(args.branch_file[0]).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]

torch.set_default_device('cuda:'+device)
batch_size = 2**14

dir_results = '/mnt/data/results/branch-project/results-indirect/test/'+benchmark+"/"
confidence_dir = "/mnt/data/results/branch-project/confidence-scores/"
dir_h5 = '/mnt/data/results/branch-project/datasets-indirect/'+benchmark+"/"

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
model.to('cuda:'+device)

def filter_instances(loader):

    #print("Filtering instances")

    #for batch_x, _, _, _ in loader:
    #    for i in range(len(batch_x)):
    #        history = batch_x[i].cpu().numpy().astype(np.int16).tobytes()
    #        if history not in history_indxs:
    #            history_list.append(history)
    #            history_indxs[history] = len(history_list) - 1

    #print("Creating data frame")

    #history_df = pl.DataFrame({
    #    "history": pl.Series("history", history_list)
    #})

    #del history_list

    #print("Collected unique histories")

    unique_histories = {}
    workload_list = []
    checkpoint_list = []
    output_list = []
    label_list = []
    history_list = []
    full_history_list = []
    for batch_x, batch_y, full_histories, checkpoints, workloads in loader:
        with torch.no_grad():
            batch_x.to('cuda:'+device)
            outputs = model(batch_x)
        for i in range(len(outputs)):
            workload = workloads[i]
            if workload not in unique_histories:
                unique_histories[workload] = {}
            checkpoint = int(checkpoints[i])
            if checkpoint not in unique_histories[workload]:
                unique_histories[workload][checkpoint] = defaultdict(int)
            history = batch_x[i].cpu().numpy().astype(np.int16)
            output = outputs[i].cpu()
            label = batch_y[i].cpu()
            if ((output > 0 and label == 1) or (output < 0 and label == 0)):
                unique_histories[workload][checkpoint][history.tobytes()] += 1
                if unique_histories[workload][checkpoint][history.tobytes()] > 1: continue
                workload_list.append(workload)
                checkpoint_list.append(checkpoint)
                history_list.append(history)
                output_list.append(float(output))
                label_list.append(int(label))
                full_history_list.append(full_histories[i].cpu().numpy().astype(np.int64))

    print("Ran inferences")

    weights = []
    for i in range(len(workload_list)):
        workload = workload_list[i]
        checkpoint = checkpoint_list[i]
        history = history_list[i]
        total = sum(unique_histories[workload][checkpoint].values())
        weights.append(unique_histories[workload][checkpoint][history.tobytes()]/total)

    df = pl.DataFrame({
        "workload": np.array(workload_list),
        "checkpoint": np.array(checkpoint_list, dtype=np.uint8),
        "label": np.array(label_list, dtype=np.uint8),
        "output": np.array(output_list),
        "history": np.array(history_list, dtype=np.int16),
        "full_history": np.array(full_history_list),
        "weight": np.array(weights)
    })

    return df

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt, map_location=torch.device('cuda:'+device)))
    model.to('cuda:'+device)
    model.eval()
 
    train_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'train')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    eval_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'validate')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    print("Num train instances: ", len(train_loader))
    print("Num eval instances: ", len(eval_loader))
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=batch_size, shuffle=False)

    #print("Running train batches: ", len(train_loader))
    #train_confidences = filter_instances(train_loader)
    del train_loader
    print("Running eval batches: ", len(eval_loader))
    eval_confidences = filter_instances(eval_loader)
    del eval_loader

    #pl.concat([train_confidences, eval_confidences])
    #pl.concat([train_histories, eval_histories])

    eval_confidences.write_parquet(confidence_dir+"{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark,branch,run_type))

    #test_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'test')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    #test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False)
    #print("Running test batches: ", len(test_loader))
    #test_confidences = filter_instances(test_loader)
    #del test_loader
    #test_confidences.write_parquet(confidence_dir+"{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark,branch,run_type))

    del train_confidences, eval_confidences
    torch.cuda.empty_cache()
