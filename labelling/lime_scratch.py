import yaml
import torch

dir_results = "/mnt/data/results/branch-project/results-x86/test/641.leela_s"
dir_h5 = "/mnt/data/results/branch-project/datasets-x86/641.leela_s"
good_branches = [
    0x417544,
    0x417613,
    0x4175DD,
    0x4173B8,
    0x4172EE,
    0x415322,
    0x41758C,
    0x4175C2,
    0x41774F,
    0x417306,
    0x4172FA,
    0x417312,
    0x41732A,
    0x4171B7,
    0x417762,
    0x41731E,
    0x40B883,
    0x417652,
    0x4175F8,
    0x41763F,
    0x40B614,
    0x417571,
    0x41534B,
    0x4175A7,
    0x417336,
    0x41762A,
    0x415398,
    0x415374,
    0x4177B1,
    0x41539D,
    0x417051,
    0x41531D,
    0x4178D4,
    0x417735,
    0x41834E,
    0x40B853,
    0x41737C,
    0x40B4EF,
    0x41821A,
    0x40B97F,
    0x41734B,
    0x4173D4,
    0x40B629,
    0x415759,
    0x41536F,
    0x40B9B4,
    0x40B5D0,
    0x40B8FB,
    0x413748,
    0x416F53,
    0x415346,
    0x40B87E,
    0x40B9AC,
    0x4136BD,
    0x41556F,
    0x413760,
    0x4177F1,
    0x40B95F,
    0x40B8ED,
    0x40B5FC,
    0x40B924,
    0x40B412,
    0x413754,
    0x4177DA,
    0x40B5E6,
    0x415504,
    0x415499,
    0x413707,
    0x4137AC,
    0x40B968,
    0x40B73C,
    0x40B6B3,
    0x40B932,
    0x41822E,
    0x41542C,
    0x40B4A1,
    0x41790B,
    0x40B96D,
    0x40B793,
    0x40B63E,
    0x40B8F6,
    0x40B888,
    0x4136FD,
    0x415722,
    0x40B92D,
    0x4182AF,
    0x40BA20,
    0x418229,
]

USE_CUDA = False

import os
import sys

sys.path.append(dir_results)
sys.path.append(os.getcwd())  # which means the script should be run at labelling/

from model import BranchNet
from model import BranchNetTrainingPhaseKnobs
from benchmark_branch_loader import BenchmarkBranchLoader

dir_ckpt = dir_results + "/checkpoints"
dir_config = dir_results + "/config.yaml"

with open(dir_config, "r") as f:
    config = yaml.safe_load(f)

training_phase_knobs = BranchNetTrainingPhaseKnobs()
model = BranchNet(config, training_phase_knobs)
if USE_CUDA:
    model.to("cuda")

for good_branch in good_branches:
    # Load the model checkpoint
    good_branch = hex(int(str(0x40B9B4)))
    dir_ckpt = (
        dir_results + "/checkpoints/" + "base_{}_checkpoint.pt".format(good_branch)
    )
    print("Loading model from:", dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt))
    model.eval()

    loader = BenchmarkBranchLoader("641.leela_s", good_branch, "test")
    print("Branch:", good_branch)
    print("Instances:", len(loader))
    history, label = loader.get_instance(0)
    print("Example:", history, label)

    with torch.no_grad():
        history = history.unsqueeze(0)
        if USE_CUDA:
            history = history.cuda()
            label = label.cuda()
        output = model(history)
        probs = torch.sigmoid(output)

        print("Model output:", probs)
