import sys
f = open(sys.argv[1])
lines = f.readlines()
f.close()

addrs = []
for line in lines:
    addr = line.strip().split(":")[0]
    addrs.append(addr)

f = open(sys.argv[1], "w")
for addr in addrs:
    f.write(addr+" 1\n")
f.close()
