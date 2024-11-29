import sys

simpt_file = open(sys.argv[1]).readlines()
weight_file = open(sys.argv[2]).readlines()
last_simpt = int(sys.argv[3]) #0 indexed
interval = 100e6
warmup = 10e6

simpts = []
indxs = {}
for line in simpt_file:
    interval_count = int(line.split()[0])
    simpts.append(interval_count)
    indxs[interval_count] = line.split()[1]
simpts.sort()

weights = [l.split()[0] for l in weight_file]

offset = simpts[last_simpt]*interval - warmup
print("Offset: ", offset)

adjusted_simpts = [((s*interval - offset)/interval, indxs[s]) for s in simpts[last_simpt+1:]]

adjusted_simpt_file = open(sys.argv[1]+".adjusted", "w")
adjusted_weight_file = open(sys.argv[2]+".adjusted", "w")
for c, simpt_indx in enumerate(adjusted_simpts):
    simpt, indx = simpt_indx
    adjusted_simpt_file.write(str(simpt)+" "+str(c)+"\n")
    weight = weights[int(indx)]
    adjusted_weight_file.write(weight+" "+str(c)+"\n")
adjusted_simpt_file.close()
adjusted_weight_file.close()
