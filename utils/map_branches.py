import sys

f = [i.strip() for i in open(sys.argv[1]).readlines()]
stats = open("/mnt/data/results/branch-project/stats-x86/tagescl64/641.leela_s/641.leela_s.0.2.csv").readlines()[2:]
h2p = sys.argv[2]

addrs = []
for line in stats:
    addr = int(line.split(',')[0],16)
    addrs.append(addr)

seen = set()
correlated = []
for line in f:
    addr = int(line, 16)
    if addr in seen: continue
    seen.add(addr) 
    for stat_addr in addrs:
        if addr == (stat_addr & (2**12 - 1)):
            correlated.append(stat_addr)

c = open("./correlated", "w")
for addr in correlated:
    c.write(hex(int(addr))+" 1\n")
c.write(h2p+" 1\n")
c.close()
