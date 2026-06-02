
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
import pyarrow as pa
import pyarrow.parquet as pq
 
parser = argparse.ArgumentParser(prog='filter_instances', description='run lime forever and ever')
 
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
    good_branches = [i.strip() for i in open(args.branch_file).readlines()[0].split(",")]
else:
    good_branches = [i.strip() for i in open(benchmark+"_branches").readlines()[0].split(",")]
 
torch.set_default_device('cuda:'+device)
batch_size = 2**14
 
workdir = "/mnt/datasets/lp721/"
dir_results = workdir+"/results/test/"+benchmark+"/"
confidence_dir = workdir+"/confidence-scores/"
dir_h5 = workdir+"datasets/"+benchmark+"/"
 
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
 
def filter_instances(loader, output_path):
    """Run inference over ``loader`` and stream the correctly-classified
    instances straight to ``output_path`` as a parquet file.
 
    Each batch is converted to an Arrow table and written immediately via a
    lazily-initialized ParquetWriter (same disk-streaming approach as
    explain_instances_streamed.py's ``writer``), so we only ever hold one
    batch's worth of rows in memory instead of the entire result set.
    """
 
    writer = None
    total_written = 0
 
    try:
        for batch_x, batch_y, full_histories, checkpoints, workloads in loader:
            with torch.no_grad():
                batch_x.to('cuda:'+device)
                outputs = model(batch_x)
 
            # Per-batch buffers only - flushed to disk at the end of each batch.
            workload_list = []
            checkpoint_list = []
            output_list = []
            label_list = []
            history_list = []
            full_history_list = []
 
            for i in range(len(outputs)):
                workload = workloads[i]
                checkpoint = int(checkpoints[i])
                history = batch_x[i].cpu().numpy().astype(np.int16)
                output = outputs[i].cpu()
                label = batch_y[i].cpu()
                if ((output > 0 and label == 1) or (output < 0 and label == 0)):
                    workload_list.append(workload)
                    checkpoint_list.append(checkpoint)
                    history_list.append(history)
                    output_list.append(float(output))
                    label_list.append(int(label))
                    full_history_list.append(full_histories[i].cpu().numpy().astype(np.int64))
 
            # Nothing in this batch passed the confidence filter.
            if not output_list:
                continue
 
            df = pl.DataFrame({
                "workload": np.array(workload_list),
                "checkpoint": np.array(checkpoint_list, dtype=np.uint8),
                "label": np.array(label_list, dtype=np.uint8),
                "output": np.array(output_list),
                "history": np.array(history_list, dtype=np.int16),
                "full_history": np.array(full_history_list),
            })
 
            # Stream this batch to disk. The ParquetWriter is created lazily
            # from the first batch's schema and reused for every subsequent
            # batch, then closed in the finally block below.
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_written += table.num_rows
 
            del df, table
 
    finally:
        if writer is not None:
            writer.close()
 
    print("Ran inferences")
 
    if writer is None:
        print("Warning: no instances passed the filter; no parquet written to", output_path)
 
    return total_written
 
for branch in good_branches:
 
    print('Branch:', branch)
 
    # Load the model checkpoint
    dir_ckpt = dir_results + '/checkpoints/' + 'base_{}_checkpoint.pt'.format(branch)
    print('Loading model from:', dir_ckpt)
    model.load_state_dict(torch.load(dir_ckpt, map_location=torch.device('cuda:'+device)))
    model.to('cuda:'+device)
    model.eval()
 
    #train_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'train')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    eval_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'validate')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    #print("Num train instances: ", len(train_loader))
    print("Num eval instances: ", len(eval_loader))
    #train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=batch_size, shuffle=False)
 
    eval_output_path = confidence_dir + "{}_branch_{}_{}_confidences_filtered.parquet".format(benchmark, branch, run_type)
 
    #print("Running train batches: ", len(train_loader))
    #train_output_path = confidence_dir + "{}_branch_{}_{}_confidences_filtered_train.parquet".format(benchmark, branch, run_type)
    #filter_instances(train_loader, train_output_path)
    #del train_loader
 
    print("Running eval batches: ", len(eval_loader))
    print("Writing to " + eval_output_path)
    num_eval = filter_instances(eval_loader, eval_output_path)
    print("Wrote {} eval instances".format(num_eval))
    del eval_loader
 
    #test_loader = BranchDataset([dir_h5+p for p in get_traces.get_hdf5_set(benchmark, 'test')], int(branch,16), config['history_lengths'][-1], config['pc_bits'], config['pc_hash_bits'], config['hash_dir_with_pc'])
    #test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False)
    #print("Running test batches: ", len(test_loader))
    #test_output_path = confidence_dir + "{}_branch_{}_{}_confidences_filtered_test.parquet".format(benchmark, branch, run_type)
    #filter_instances(test_loader, test_output_path)
    #del test_loader
 
    torch.cuda.empty_cache()
