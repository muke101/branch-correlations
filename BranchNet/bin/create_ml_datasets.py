#!/usr/bin/env python3
'''
This script converts a sequential binary trace of branches produced by the 
tracer PIN tool to a HDF5 dataset and marks the hard to predict branches in
the trace. This format is more suitable for random accesses to arbitrary
instances of hard-to-predict branches during training.
'''

import bz2
import h5py
import multiprocessing
import numpy as np
import os
import struct
import polars as pl

import common
from common import PATHS, BENCHMARKS_INFO

#TARGET_BENCHMARKS = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
# "625.x264_s", "631.deepsjeng_s",
# "641.leela_s", "657.xz_s", "602.gcc_s",
# "620.omnetpp_s", "648.exchange2_s"]
TARGET_BENCHMARKS = ["605.mcf_s", "623.xalancbmk_s",
 "625.x264_s", "631.deepsjeng_s",
 "641.leela_s", 
 "620.omnetpp_s", "648.exchange2_s"]
TARGET_BENCHMARKS = ["641.leela_s"]
HARD_BRS_FILE = 'top100'
NUM_THREADS = 32
PC_BITS = 30


def read_full_branch_trace(trace_path):
    df = pl.read_parquet(trace_path)
    return df['inst_addr'].to_numpy(), df['taken'].to_numpy() 

def read_warmed_up_branch_trace(trace_path):
    df = pl.read_parquet(trace_path)
    df = df.filter(df['warmed_up'] == 1)
    return df['inst_addr'].to_numpy(), df['taken'].to_numpy() 

def create_new_dataset(dataset_path, pcs, directions):
    '''
    Create a new hdf5 file and copy over the history to the file.
    Branch PCs and directions are concatenated. Only the least significant bits of PC
    (controlled by PC_BITS) are stored.
    '''
    stew_bits = PC_BITS + 1
    if stew_bits < 8:
        stew_dtype = np.uint8
    elif stew_bits < 16:
        stew_dtype = np.uint16
    elif stew_bits < 32:
        stew_dtype = np.uint32
    elif stew_bits < 64:
        stew_dtype = np.uint64
    else:
        assert False, 'History elements of larger than 64 bits are not supported'

    pc_mask = (1 << PC_BITS) - 1
    fptr = h5py.File(dataset_path, 'w')
    processed_history = ((pcs & pc_mask) << 1) | directions
    processed_history = processed_history.astype(stew_dtype)
    fptr.attrs['pc_bits'] = PC_BITS
    fptr.create_dataset(
        "history",
        data=processed_history,
        compression='gzip',
        compression_opts=9,
    )
    full_history = (pcs << 1) | directions
    full_history = full_history.astype(np.uint64)
    fptr.create_dataset(
        "full_history",
        data=full_history,
        compression='gzip',
        compression_opts=9,
    )
    return fptr

def get_work_items():
    work_items = []    
    for benchmark in TARGET_BENCHMARKS:
        hard_brs = common.read_hard_brs(benchmark, HARD_BRS_FILE)
        #traces_dir = '{}/{}'.format(PATHS['branch_traces_dir'], benchmark)
        traces_dir = PATHS['branch_traces_dir']
        datasets_dir = '{}/{}'.format(PATHS['ml_datasets_dir'], benchmark)
        os.makedirs(datasets_dir, exist_ok=True)
        for inp_info in BENCHMARKS_INFO[benchmark]['inputs']:
            for simpoint_info in inp_info['simpoints']:
                file_basename = '{}.{}.{}'.format(
                    benchmark, inp_info['name'], simpoint_info['id']+1)
                trace_path = '{}/{}.trace'.format(
                    traces_dir, file_basename)
                dataset_path = '{}/{}.hdf5'.format(
                    datasets_dir, file_basename)
                
                #if os.path.exists(dataset_path): continue
                work_items.append((trace_path, dataset_path, hard_brs))
    return work_items


def gen_dataset(trace_path, dataset_path, hard_brs):
    print('reading file', trace_path)
    pcs, directions = read_full_branch_trace(trace_path)

    fptr = create_new_dataset(dataset_path, pcs, directions)

    #only index along the part of the trace after warmup
    warmed_up_pcs, warmed_up_directions = read_warmed_up_branch_trace(trace_path)

    for br_pc in hard_brs:
        #print('processing branch {}'.format(hex(br_pc)))
        #find indicies of hard branches
        trace_br_indices = np.argwhere(pcs == br_pc).squeeze(axis=1)
        trace_br_indices = np.array([i for i in filter(lambda i: i > len(pcs) - len(warmed_up_pcs), trace_br_indices)])
        fptr.create_dataset(
            'br_indices_{}'.format(hex(br_pc)),
            data=trace_br_indices,
            compression='gzip',
            compression_opts=9,
        )
        num_taken = np.count_nonzero(
            np.bitwise_and(warmed_up_pcs == br_pc, warmed_up_directions == 1))
        num_not_taken = np.count_nonzero(
            np.bitwise_and(warmed_up_pcs == br_pc, warmed_up_directions == 0))
        fptr.attrs['num_taken_{}'.format(hex(br_pc))] = num_taken
        fptr.attrs['num_not_taken_{}'.format(hex(br_pc))] = num_not_taken

    fptr.close()
    print('Created output file', dataset_path)


def main():
    work_items = get_work_items()
    with multiprocessing.Pool(NUM_THREADS) as pool:
        pool.starmap(gen_dataset, work_items)


if __name__ == '__main__':
    main()
