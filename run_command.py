import os
import sys 
from get_instances import get_instances

pem_file = "rlresearch.pem"

instances = get_instances()
worker_instances = instances["Workers"]
command = sys.argv[1]

for instance in worker_instances:
	print(f"Running command {command} on instance {instance}")
	os.system(f'ssh -i {pem_file} -o StrictHostKeyChecking=no ubuntu@{instance} "{command}"')
