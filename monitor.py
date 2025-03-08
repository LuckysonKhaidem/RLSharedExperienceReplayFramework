import subprocess
import time
from get_instances import get_instances
import boto3

pem_file = "rlresearch.pem"

instances = get_instances()
worker_instances = instances["Workers"]
redis_instance = instances["Redis"]

instance_command = "pgrep python"

while True:
    done_instances = 0
    for instance in worker_instances:
        result = subprocess.run(f'ssh -i {pem_file} -o StrictHostKeyChecking=no ubuntu@{instance} "{instance_command}"', capture_output=True, text=True, shell=True)
        print(f"result from instance {instance} = {result.stdout}")
        if (result.stdout == ""):
            done_instances += 1
    
    if done_instances >= len(worker_instances):
        ec2_client = boto3.client('ec2', region_name = "us-west-2")
        
        print("stopping all instances")
        ec2_client.stop_instances(InstanceIds = instances["instance_ids"])
        exit()
    
    time.sleep(60 * 5)


	
