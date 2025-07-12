import subprocess
import os

start = 0.001
end = 0.009
step = 0.001

n_values1 = [round(start + i * step, 6) for i in range(int((end - start) / step) + 1)]
# n_values2 = [round(start2 + i * step2, 6) for i in range(int((end2 - start2) / step2) + 1)]
n_values = n_values1 
# n_values = [0.01, 0.001]

for n in n_values:
    print(f"Running scripts for n = {n}")



    os.chdir("/home/bj321/Developer/branch-transformer-addon")
    # First script
    subprocess.run([
        "python3", 
        "/home/bj321/Developer/branch-transformer-addon/aggregate_relevancy.py", 
        "brute",
        str(n)
    ])

    os.chdir("/home/bj321/Developer/gem5-new")

    # Second script
    subprocess.run([
        "python3", 
        "run_models.py", 
        "--run-type", f"full_h2p_brute_force_{n}", 
        "--addr-types", "base", 
        "--benches", "641.leela_s", 

        "--cpu-models", "m4"
    ])