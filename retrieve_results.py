import os
import sys 

instances = sys.argv[1].split(",") 
pem_file = "rlresearch.pem"
src_file = sys.argv[2]
num_instances = len(instances)
for i in range(num_instances):
	os.system(f"mkdir -p results/instance{i}")

for i,instance in enumerate(instances):
	print(f"Retreiving results from instance {instance}")
	os.system(f'scp -i {pem_file} ubuntu@{instance}:{src_file} ./results/instance{i}')
