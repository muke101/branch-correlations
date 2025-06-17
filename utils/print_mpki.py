import get_traces
import polars as pl

num_insts = 100e6

total_mispreds = 0
gem5_mispreds = [875835, 1192166, 1144298, 1294464, 1338324, 1302091, 1301781, 1201894]
total_gem5_mispreds = 0
c = 0
for trace,weight in get_traces.get_trace_set("641.leela_s", "test"):
    df = pl.read_parquet(get_traces.trace_dir+trace)
    mispreds = df['mispredicted'].sum()
    total_mispreds += mispreds*weight
    total_gem5_mispreds += gem5_mispreds[c]*weight
    print("Simpoint "+trace.split('.trace')[0]+" has MPKI: "+str(1000*mispreds/num_insts))
    c+=1
print("Weighted MPKI: "+str(1000*total_mispreds/num_insts))
print("Weighted mispredics: "+str(total_mispreds))
print("Weighted gem5 mispredics: "+str(total_gem5_mispreds))
