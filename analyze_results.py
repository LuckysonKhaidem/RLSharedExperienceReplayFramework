import json
from matplotlib import pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

n_instances = 5
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def main():
    env_name = sys.argv[1]
    multi_data = []
    for i in range(n_instances):
        try:
            with open(f"results/instance{i}/rewards_{env_name}.json") as f:
                multi_data.append(json.loads(f.read()))
        except:
            continue
    with open(f"rewards_{env_name}_local.json") as f:
        single_data = json.loads(f.read())
    
    n = min(len(multi_data[0]), len(single_data))
    print(f"Length {len(single_data)} {len(multi_data[0])}")
    for i in range(len(multi_data)):
        multi_data[i] = multi_data[i][:n]
    single_data = single_data[:n]

    plt.plot(moving_average(single_data, 100), c = "black", label = "Single DDQN Agent")
    for i in range(len(multi_data)):
        plt.plot(moving_average(multi_data[i], 100), label = f"Async DDQN Agent{i}")
    plt.xlabel("Episodes")
    plt.ylabel("Moving average reward")
    plt.title(f"Convergence Comparision on {env_name} environment")
    plt.legend()
    plt.savefig(f"{env_name}.pdf")
if __name__ == "__main__":
    main()
