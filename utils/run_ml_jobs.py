#!/usr/bin/env python3

from collections import namedtuple
import subprocess
import multiprocessing
import glob
import os
import shutil
import sys
import get_traces

benches = ["600.perlbench_s", "605.mcf_s", "623.xalancbmk_s",
           "625.x264_s", "631.deepsjeng_s",
           "641.leela_s", "657.xz_s", "602.gcc_s",
           "620.omnetpp_s", "648.exchange2_s"]
run_type = sys.argv[1]
results_dir = "/mnt/data/results/branch-project/branchnet-results/"+run_type
branchnet_dir = "/work/muke/Branch-Correlations/BranchNet/src/branchnet/"

batch_size = 2048
training_steps = [100, 100, 100]
fine_tuning_steps = [50, 50, 50]
learning_rate = 0.1
lasso_coefficient = 0.0
regularization_coefficient = 0.0
cuda_device = 0
log_validation = False

create_workdirs = True
workdirs_override_ok = True

def create_run_command(workdir, training_datasets, evaluation_datasets,
                       validation_datasets, br_pc, training_mode):
    return ('cd {workdir}; python run.py '
            '-trtr {tr} -evtr {ev} -vvtr {vv} --br_pc {pc} --batch_size {batch} '
            '-bsteps {bsteps} -fsteps {fsteps} --log_progress {log_validation} '
            '-lr {lr} -gcoeff {gcoeff} -rcoeff {rcoeff} -mode {mode} '
            '-c config.yaml --cuda_device {cuda} &> run_logs/{pc}.out'.format(
                workdir=workdir,
                tr=' '.join(training_datasets),
                ev=' '.join(evaluation_datasets),
                vv=' '.join(validation_datasets),
                pc=hex(br_pc),
                batch=batch_size,
                bsteps=' '.join(map(str, training_steps)),
                fsteps=' '.join(map(str, fine_tuning_steps)),
                log_validation='--log_validation' if log_validation else '',
                lr=learning_rate,
                gcoeff=lasso_coefficient,
                rcoeff=regularization_coefficient,
                mode=training_mode,
                cuda=cuda_device,
            ))

def create_workdirs():
    for bench in benches:
        workdir = results_dir+"/"+bench
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + '/run_logs', exist_ok=True)
        os.makedirs(workdir + '/visual_logs', exist_ok=True)
        for filename in ['run.py', 'dataset_loader.py', 'model.py']:
            shutil.copy(branchnet_dir + '/' + filename, workdir)
        shutil.copyfile('{}/configs/{}.yaml'.format(branchnet_dir, 'big'), workdir + '/config.yaml')

def create_job_commands():
    cmds = []
    for bench in benches:
        workdir = results_dir+"/"+bench
        hard_brs_file = open("/mnt/data/results/branch-project/h2ps/validate/"+bench, 'r')
        hard_brs = [int(pc,16) for pc in hard_brs_file.readlines()]
        hard_brs_file.close()
        training_datasets = get_traces.get_hdf5_set(bench, 'train')
        evaluation_datasets = get_traces.get_hdf5_set(bench, 'test')
        validation_datasets = get_traces.get_hdf5_set(bench, 'validate')

        for br in hard_brs:
            cmd = create_run_command(workdir, training_datasets, evaluation_datasets,
                                     validation_datasets, br, 'float')
            cmds.append(cmd)

    return cmds

def run_cmd_using_shell(cmd):
  print('Running cmd:', cmd)
  subprocess.call(cmd, shell=True)

def main():
    if create_workdirs:
        create_workdirs()

    cmds = create_job_commands()
    print(cmds)
    exit(0)
    with multiprocessing.Pool(28) as pool:
        pool.map(run_cmd_using_shell, cmds)


if __name__ == '__main__':
    main()
