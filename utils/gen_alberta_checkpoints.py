import subprocess
from subprocess import Popen
import sys
import argparse
import os

spec_path = "/work/muke/spec2017-expanded/"
gem5 = "/work/muke/PND-Loads/gem5-gen/"
workloads = "/work/muke/alberta-workloads/"

parser = argparse.ArgumentParser(prog='gen_expanded_checkpoints', description='')

parser.add_argument('--bench', type=str, required=True)
args = parser.parse_args()

bench = args.bench.split(',')[0]
checkpoint_path = "/work/muke/checkpoints-expanded/"+bench

os.chdir(spec_path+"benchspec/CPU/"+bench+"/run/run_peak_refspeed_mytest-64.0000")
stripped_name = bench.split('.')[1].split('_')[0]
