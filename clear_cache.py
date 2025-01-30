import redis
from get_instances import get_instances
import sys

instances = get_instances()
target_key = sys.argv[1]
redis_instance = instances["Redis"]
redis_client = redis.Redis(host = redis_instance, port=6379)

print(f"Deleting key '{target_key}' with value '{redis_client.get(target_key)}'")
redis_client.delete(target_key)

redis_client.close()
