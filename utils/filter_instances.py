import yaml
import torch
import get_traces
from benchmark_branch_loader import BenchmarkBranchLoader
import os
import sys
benchmark = sys.argv[1]
dir_results = '/mnt/data/results/branch-project/results/test/648.exchange2_s'
dir_h5 = '/mnt/data/results/branch-project/datasets/648.exchange2_s'
sys.path.append(dir_results)
sys.path.append(os.getcwd())
from model import BranchNet
from model import BranchNetTrainingPhaseKnobs
