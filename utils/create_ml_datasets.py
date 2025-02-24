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

PC_BITS = 30

benches = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
           "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "648.exchange2_s"]

hard_branches_dir = "/mnt/data/results/branch-project/h2ps/"

def read_branch_trace(trace_path):
    df = pl.read_parquet(trace_path)
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
    fptr = h5py.File(dataset_path, 'w-')
    processed_history = ((pcs & pc_mask) << 1) | directions
    processed_history = processed_history.astype(stew_dtype)
    fptr.attrs['pc_bits'] = PC_BITS
    fptr.create_dataset(
        "history",
        data=processed_history,
        compression='gzip',
        compression_opts=9,
    )
    return fptr

def get_work_items():
    work_items = []    
    for bench in benches:
        traces = [t[0] for t in get_traces.get_trace_set(bench, "test")]
        traces += [t[0] for t in get_traces.get_trace_set(bench, "train")]
        traces += [t[0] for t in get_traces.get_trace_set(bench, "validate")]
        hard_brs_file = open(hard_branches_dir+bench, "r")
        hard_brs = [int(pc,16) for pc in hard_brs_file.readlines()]
        hard_brs_file.close()
        for trace in traces:
            trace_path = get_traces.trace_dir+trace
            dataset_path = get_traces.hdf5_dir+trace.split(".trace")[0]+'.hdf5'
            work_items.append((trace_path, dataset_path, hard_brs))
    return work_items


def gen_dataset(trace_path, dataset_path, hard_brs):
    print('reading file', trace_path)
    pcs, directions = read_branch_trace(trace_path)

    print('Creating output file', dataset_path)
    fptr = create_new_dataset(dataset_path, pcs, directions)

    for br_pc in hard_brs:
        print('processing branch {}'.format(hex(br_pc)))
        #find indicies of hard branches
        trace_br_indices = np.argwhere(pcs == br_pc).squeeze(axis=1)
        fptr.create_dataset(
            'br_indices_{}'.format(hex(br_pc)),
            data=trace_br_indices,
            compression='gzip',
            compression_opts=9,
        )
        num_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 1))
        num_not_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 0))
        fptr.attrs['num_taken_{}'.format(hex(br_pc))] = num_taken
        fptr.attrs['num_not_taken_{}'.format(hex(br_pc))] = num_not_taken


def main():
    work_items = get_work_items()
    with multiprocessing.Pool(16) as pool:
        pool.starmap(gen_dataset, work_items)

if __name__ == '__main__':
    main()
    # trace_path = "/mnt/data/results/branch-project/traces/625.x264_s.notld-2.3.trace"
    # f = open("test_pcs")
    # hard_brs = [int(x,16) for x in f.read().splitlines()]
    # gen_dataset(trace_path, "test.h5", hard_brs)
