import os
import sys 

instances = sys.argv[1].split(",") 
pem_file = "rlresearch.pem"
command = sys.argv[2]

for instance in instances:
	print(f"Starting instance {instance}")
	os.system(f'ssh -i {pem_file} -o StrictHostKeyChecking=no ubuntu@{instance} "{command}"')

print("Started all instances")
