import csv
import sys
import re
import io
import itertools

def filtered_input(stream):
    started = False
    for line in stream:
        if started:
            yield line
        elif "up!" in line:
            print("Tracing started")
            started = True

filtered_stream = filtered_input(sys.stdin)

accuracies = {}

for line in filtered_stream:
    if "PREDICTION" in line:
        addr = int(line.split("PREDICTION:")[1], 16)
        if addr not in accuracies:
            accuracies[addr] = [0,0] #total, incorrect
        accuracies[addr][0] += 1
    elif "MISPREDICT" in line:
        addr = int(line.split("MISPREDICT:")[1], 16)
        if addr in accuracies: accuracies[addr][1] += 1 #might be a misprediction from a prediction made during warmup, in which case it won't exist in the dictionary yet

with open(sys.argv[1], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Addr", "Total", "Incorrect"])
    for addr, (total, incorrect) in accuracies.items():
        writer.writerow([hex(addr), total, incorrect])

