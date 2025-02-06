import os
import sys 
from get_instances import get_instances

instances = get_instances()
worker_instances = instances["Workers"]

command = sys.argv[1]
pem_file = "rlresearch.pem"
src_file = sys.argv[1]
num_instances = len(instances)

for i in range(num_instances):
	os.system(f"mkdir -p results/instance{i}")

for i,instance in enumerate(worker_instances):
	print(f"Retreiving results from instance {instance}")
	os.system(f'scp -i {pem_file} ubuntu@{instance}:{src_file} ./results/instance{i}')
