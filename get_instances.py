import boto3
from pprint import pprint




def get_instances():
    ec2_client = boto3.client('ec2', region_name = "us-west-2")
    result = ec2_client.describe_instances()
    command_result = {}
    for data in result["Reservations"]:
        for instance in data["Instances"]:
            if "State" in instance and instance["State"]["Name"] == "running":
                if "Tags" in instance:
                    for tag in instance["Tags"]:
                        if tag["Key"] == "Name":
                            if tag["Value"] == "Redis":
                                command_result["Redis"] = instance["PublicIpAddress"]
                            else:
                                if "Workers" not in command_result:
                                    command_result["Workers"] = []
                                command_result["Workers"].append(instance["PublicIpAddress"])
            if "instance_ids" not in command_result:
                command_result["instance_ids"] = []
            
            command_result["instance_ids"].append(instance["InstanceId"])
    print(",".join(command_result["instance_ids"]))
    return command_result

if __name__ == "__main__":
    print(get_instances())