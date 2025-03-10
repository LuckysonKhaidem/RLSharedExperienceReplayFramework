import json
import os
from collections import defaultdict
with open("MountainCar_Raw") as f:
    lines = f.read().split("\n")

d = defaultdict(list)

for line in lines:
    comp = line.split(" ")
    agent = comp[5]
    reward = float(comp[-1])
    d[agent].append(reward)

for i, agent in enumerate(d.keys()):
    rewards = d[agent]
    with open(f"results/instance{i}/rewards_Pendulum-v1.json", "w") as f:
        f.write(json.dumps(rewards))

