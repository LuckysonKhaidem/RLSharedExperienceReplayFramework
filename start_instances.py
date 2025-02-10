from get_instances import get_instances
import boto3

pem_file = "rlresearch.pem"

instances = get_instances()

ec2_client = boto3.client('ec2', region_name = "us-west-2")

ec2_client.start_instances(InstanceIds = instances["instance_ids"])
