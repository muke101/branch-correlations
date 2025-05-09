import r2pipe
import sys
import get_traces
import polars as pl

trace_dir = "/mnt/data/results/branch-project/traces/"
h2p_dir = "/mnt/data/results/branch-project/h2ps/"
binary = sys.argv[1]
r2 = r2pipe.open(binary)
h2p_file = h2p_dir+binary+"_top100"

addrs = [int(i.strip(),16) for i in open(h2p_file).readlines()]

r2.cmd("aaa")

for addr in addrs:
    inst = r2.cmdj('pdj 1 @'+str(addr))[0]
    if inst['type'] != 'cjmp':
        print("Error: H2P type not cjmp. Found: ", inst['type'])
        exit(1)
    func_addr = inst['fnc_addr']
    for func in r2.cmdj("isj"):
        if func['type'] == "FUNC" and func['vaddr'] == func_addr:
            print("H2P "+hex(inst['vaddr'])+" is in function "+func['realname'])

qsorts = [i for i in r2.cmdj("isj") if i['type'] == "FUNC" and "qsort" in i['realname']]

brs = []
for q in qsorts:
    addr = q['vaddr']
    size = str(int(r2.cmdj("pdfj @ "+addr)['size']) // 4)
    insts = r2.cmdj("pdj "+size+" @ "+addr)
    for inst in insts:
        if inst['type'] == "cjmp": brs.append(inst['vaddr'])

total = 0
weighted_total = 0
for trace, weight in get_traces.get_trace_set("641.leela_s", "test"):
    df = pl.load_parquet(trace_dir+trace)
    trace_total = 0
    for inst_addr in df['inst_addr'].unique():
        if inst_addr in brs:
            total += 1
            trace_total += 1
    weighted_total += trace_total * weight
            #print("Found qsort branch "+inst_addr+" in "+trace)
print("Number of qsort branches found in test set traces: ", total)
print("Number of weighted qsort branches found in test set traces: ", weighted_total)
