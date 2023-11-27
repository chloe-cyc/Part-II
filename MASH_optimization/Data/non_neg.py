import numpy as np
wig =np.loadtxt("/u/dem/kebl6911/Part-II/MASH_optimization/Data/pop.out")

wig[:,6]=0
time = wig[:,0]
wig = wig[:,1:8]
print(wig)

row_sums = wig.sum(axis=1, keepdims=True)
array_normalized = wig / row_sums

wig = np.column_stack((time,array_normalized))
np.savetxt(f"wignormalisedpop.dat", wig, delimiter = " ")