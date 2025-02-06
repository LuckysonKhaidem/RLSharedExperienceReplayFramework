import subprocess
import sys 
from get_instances import get_instances

pem_file = "rlresearch.pem"

instances = get_instances()
worker_instances = instances["Workers"]
redis_instance = instances["Redis"]

instance_command = "pgrep python"

for instance in worker_instances:
	print(f"Running command {instance_command} on instance {instance}")
	result = subprocess.run(f'ssh -i {pem_file} -o StrictHostKeyChecking=no ubuntu@{instance} "{instance_command}"', capture_output=True, text=True, shell=True)
	print("Output is ", result.stdout)
	
