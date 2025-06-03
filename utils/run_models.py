import os
import argparse
import subprocess
import psutil
import time

label_file_dir = "/work/muke/Branch-Correlations/label_files/"
cpu_model_dir = "/work/muke/Branch-Correlations/cpu_models/"
gem5_dir = "/work/muke/Branch-Correlations/gem5/"

parser = argparse.ArgumentParser(prog='run_models', description='run over multiple label files and cpu models')

parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--cpu-models', type=str, required=True)
parser.add_argument('--label-types', type=str, required=True)
parser.add_argument('--benches', type=str, required=False)
args = parser.parse_args()

run_type = args.run_type.split(',')[0]
label_types = args.label_types.split(',')
cpu_models = args.cpu_models.split(',')
benches = ""
if args.benches != None:
    benches = args.benches 

os.chdir(gem5_dir)

#run all models on base first to generate comparison results
if 'base' in label_types:
    for model in cpu_models:
        cp = subprocess.run("cp "+cpu_model_dir+model+".py src/cpu/o3/BaseO3CPU.py", shell=True, check=True)
        fu_config = model.split('-')[0]+"-fu.py"
        cp = subprocess.run("cp "+cpu_model_dir+fu_config+" src/cpu/o3/FuncUnitConfig.py", shell=True, check=True)
        scons = subprocess.run("scons build/ARM/gem5.fast -j 31 --with-lto --linker=gold", shell=True, check=True)
        run = subprocess.run("python3 /work/muke/PND-Loads/utils/run_all_chkpts.py "+run_type+" base "+model+" with_base "+benches , shell=True, check=True)

label_types = [a for a in label_types if a != 'base']

for model in cpu_models:
    cp = subprocess.run("cp "+cpu_model_dir+model+".py src/cpu/o3/BaseO3CPU.py", shell=True, check=True)
    fu_config = model.split('-')[0]+"-fu.py"
    cp = subprocess.run("cp "+cpu_model_dir+fu_config+" src/cpu/o3/FuncUnitConfig.py", shell=True, check=True)
    scons = subprocess.run("scons build/ARM/gem5.fast -j 31 --with-lto --linker=gold", shell=True, check=True)
    for label_type in label_types:
       run = subprocess.run("python3 /work/muke/PND-Loads/utils/run_all_chkpts.py "+run_type+" "+label_type+" "+model+" without_base "+benches, shell=True, check=True)
