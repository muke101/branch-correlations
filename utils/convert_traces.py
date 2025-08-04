import get_traces
import os
import csv
import polars as pl 
from concurrent.futures import ThreadPoolExecutor, as_completed


stats_dir = "/mnt/data/results/branch-project/stats-indirect/tagescl64/"
trace_dir = "/mnt/data/results/branch-project/traces-indirect/"

def write_stats(trace):
    print("Processing ", trace)
    benchmark = '.'.join(trace.split('.')[0:2])
    out_file = stats_dir+benchmark+"/"+trace.split('.trace')[0]+".csv"
    if os.path.exists(out_file): return
    df = pl.read_parquet(trace_dir+trace)
    records = []
    aggregate_dir_t_pred_t = 0;
    aggregate_dir_t_pred_nt = 0;
    aggregate_dir_nt_pred_t = 0;
    aggregate_dir_nt_pred_nt = 0;
    for inst_addr in df['inst_addr'].unique():
        dir_t_pred_t = 0;
        dir_t_pred_nt = 0;
        dir_nt_pred_t = 0;
        dir_nt_pred_nt = 0;
        filtered = df.filter((df['inst_addr'] == inst_addr) & (df['warmed_up'] == 1))
        total = filtered.height
        #branch only occurs in warmup
        if total == 0: continue
        total_incorrect = filtered['mispredicted'].sum()
        total_correct = total - total_incorrect
        for row in filtered.iter_rows():
            taken = int(row[1])
            mispredicted = int(row[2])
            if taken and not mispredicted:
                dir_t_pred_t += 1
                aggregate_dir_t_pred_t += 1
            if taken and mispredicted:
                dir_t_pred_nt += 1
                aggregate_dir_t_pred_nt += 1
            if not taken and not mispredicted:
                dir_nt_pred_nt += 1
                aggregate_dir_nt_pred_nt += 1
            if not taken and mispredicted:
                dir_nt_pred_t += 1
                aggregate_dir_nt_pred_t += 1
        if (dir_t_pred_t + dir_nt_pred_nt != total_correct):
            print("Error! Total correct from df doesn't match dir_t_pred_t + dir_nt_pred_nt!")
            exit(1)
        accuracy = str(100.0 * total_correct / total)+"%"
        records.append((hex(int(inst_addr)), accuracy, total_incorrect, total_correct, total, dir_t_pred_t, dir_t_pred_nt, dir_nt_pred_t, dir_nt_pred_nt))
    aggregate_correct = aggregate_dir_t_pred_t + aggregate_dir_nt_pred_nt
    aggregate_incorrect = aggregate_dir_t_pred_nt + aggregate_dir_nt_pred_t
    aggregate_total = aggregate_correct + aggregate_incorrect
    aggregate_accuracy = str(100.0 * aggregate_correct / aggregate_total) + "%"
    records = sorted(records, key=lambda x: x[2], reverse=True)
    records.insert(0, ("aggregate", aggregate_accuracy, aggregate_incorrect, aggregate_correct, aggregate_total, aggregate_dir_t_pred_t, aggregate_dir_t_pred_nt, aggregate_dir_nt_pred_t, aggregate_dir_nt_pred_nt)) 
    header = "Branch PC,Accuracy,Mispredictions,Correct Predictions,Total,dir_t_pred_t,dir_t_pred_nt,dir_nt_pred_t,dir_nt_pred_nt\n"
    f = open(out_file, "w")
    f.write(header)
    for record in records:
        for r in record[:-1]:
            f.write(str(r)+",")
        f.write(str(record[-1])+"\n")
    f.close()

if __name__ == "__main__":
    all_traces = []
    #for bench in get_traces.benchmarks:
    for bench in ["641.leela_s"]:
        if bench in ["657.xz_s", "600.perlbench_s", "602.gcc_s"]: continue
        traces = get_traces.get_trace_set(bench, 'test')
        traces += get_traces.get_trace_set(bench, 'validate')
        for trace, _ in traces:
            all_traces.append(trace)

        num_threads = os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(write_stats, trace): trace for trace in all_traces}

        for future in as_completed(futures):
            trace = futures[future]
            try:
                future.result()  # Raises exception if one occurred
            except Exception as e:
                print(f"Error processing trace {trace}: {e}")
