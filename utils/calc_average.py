import numpy as np
from scipy.stats import gmean
import sys

differences = open("differences").readlines()

if sys.argv[1] == 'cpi':


    base_cpis = []
    pnd_cpis = []
    for line in differences:
        if line.strip().startswith("Base CPI:"):
            cpi = float(line.split(':')[1])
            base_cpis.append(cpi)
        elif line.strip().startswith("PND CPI:"):
            cpi = float(line.split(':')[1])
            pnd_cpis.append(cpi)

    base_mean = gmean(base_cpis)
    pnd_mean = gmean(pnd_cpis)
    change = ((pnd_mean-base_mean)/base_mean)*100
    print("Average Base CPI: ", base_mean)
    print("Average PND CPI: ", pnd_mean)
    print("Average CPI Change: ", change)

elif sys.argv[1] == 'lookups':
    lookup_reductions = []
    for line in differences:
        if line.strip().startswith("MDPLookups"):
            reduction = abs(float(line.split(':')[1]))
            lookup_reductions.append(reduction)

    mean = np.mean(lookup_reductions)
    print("Average Lookup Reduction: ", mean)
