import get_traces
import yaml
import os
import torch
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

batch_size = 2**14

dir_results = '/mnt/data/results/branch-project/results-x86/test/'+benchmark+"/"
confidence_dir = "/mnt/data/results/branch-project/confidence-scores/"
dir_h5 = '/mnt/data/results/branch-project/datasets-x86/'+benchmark+"/"

sys.path.append(dir_results)
sys.path.append(os.getcwd())

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs

dir_ckpt = dir_results + '/checkpoints'
dir_config = dir_results + '/config.yaml'

with open(dir_config, 'r') as f:
    config = yaml.safe_load(f)

training_phase_knobs = BranchNetTrainingPhaseKnobs()

def print_instances(loader):

    unique_histories = set()
    for batch_x, batch_y, checkpoints, workloads in loader:
        for i in range(len(batch_x)):
            history = batch_x[i].cpu().numpy().astype(np.int16).tobytes()
            unique_histories.add(history)
    addrs = set()
    for hist in unique_histories:
        history = np.frombuffer(hist, dtype=np.int16)
        for addr in history: addrs.add(int(addr) & (2**12-1))
    f = open("branches", "w")
    for addr in addrs:
        f.write(hex(addr)+"\n")
    f.close()

for branch in good_branches:

    print('Branch:', branch)

    # Load the model checkpoint
 
    #train_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'train')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    #eval_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'validate')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    #print("Num train instances: ", len(train_loader))
    #print("Num eval instances: ", len(eval_loader))
    #train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    #eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=batch_size, shuffle=False)

    #print("Running train batches: ", len(train_loader))
    #train_confidences = filter_instances(train_loader)
    #del train_loader
    #print("Running eval batches: ", len(eval_loader))
    #eval_confidences = filter_instances(eval_loader)
    #del eval_loader

    #pl.concat([train_confidences, eval_confidences])
    #pl.concat([train_histories, eval_histories])

    #train_confidences.write_parquet(confidence_dir+"{}_branch_{}_confidences_filtered.parquet".format(benchmark,branch))

    test_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'test')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False)
    print("Running test batches: ", len(test_loader))
    test_confidences = print_instances(test_loader)
