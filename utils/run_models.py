import os
import argparse
import subprocess
import psutil
import time

addr_file_dir = "/work/muke/PND-Loads/addr_files/"
cpu_model_dir = "/work/muke/PND-Loads/cpu_models/"
gem5_dir = "/work/muke/PND-Loads/gem5/"

parser = argparse.ArgumentParser(prog='run_models', description='run over multiple addr files and cpu models')

parser.add_argument('--run-type', type=str, required=True)
parser.add_argument('--addr-types', type=str, required=True)
parser.add_argument('--cpu-models', type=str, required=True)
parser.add_argument('--benches', type=str, required=False)
args = parser.parse_args()

run_type = args.run_type.split(',')[0]
addr_types = args.addr_types.split(',')
cpu_models = args.cpu_models.split(',')
benches = ""
if args.benches != None:
    benches = args.benches 

if addr_types[0] == 'all': addr_types = [addr_type for addr_type in os.listdir(addr_file_dir)]
elif addr_types[0] == 'all-with-base': addr_types = [addr_type for addr_type in os.listdir(addr_file_dir)] + ['base']
if cpu_models[0] == 'all': cpu_models = [model.split('.')[0] for model in os.listdir(cpu_moddel_dir)]

os.chdir(gem5_dir)

#run all models on base first to generate comparison results
if 'base' in addr_types:
    for model in cpu_models:
        cp = subprocess.run("cp "+cpu_model_dir+model+".py src/cpu/o3/BaseO3CPU.py", shell=True, check=True)
        fu_config = model.split('-')[0]+"-fu.py"
        cp = subprocess.run("cp "+cpu_model_dir+fu_config+" src/cpu/o3/FuncUnitConfig.py", shell=True, check=True)
        scons = subprocess.run("scons build/ARM/gem5.fast -j 31 --with-lto", shell=True, check=True)
        run = subprocess.run("python3 /work/muke/PND-Loads/utils/run_all_chkpts.py "+run_type+" base "+model+" with_base "+benches , shell=True, check=True)

addr_types = [a for a in addr_types if a != 'base']

for model in cpu_models:
    cp = subprocess.run("cp "+cpu_model_dir+model+".py src/cpu/o3/BaseO3CPU.py", shell=True, check=True)
    fu_config = model.split('-')[0]+"-fu.py"
    cp = subprocess.run("cp "+cpu_model_dir+fu_config+" src/cpu/o3/FuncUnitConfig.py", shell=True, check=True)
    scons = subprocess.run("scons build/ARM/gem5.fast -j 31 --with-lto", shell=True, check=True)
    for addr_type in addr_types:
       run = subprocess.run("python3 /work/muke/PND-Loads/utils/run_all_chkpts.py "+run_type+" "+addr_type+" "+model+" without_base "+benches, shell=True, check=True)
