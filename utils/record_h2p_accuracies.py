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
        addr = int(line.split(",")[1])
        mispredicted = int(line.split(",")[2])
        if addr not in accuracies:
            accuracies[addr] = [0,0] #total, incorrect
        accuracies[addr][0] += 1
        if mispredicted:
            accuracies[addr][1] += 1

with open(sys.argv[1], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Addr", "Total", "Incorrect", "Accuracy"])
    for addr, (total, incorrect) in accuracies.items():
        writer.writerow([hex(addr), total, incorrect, (total-incorrect)/total])

