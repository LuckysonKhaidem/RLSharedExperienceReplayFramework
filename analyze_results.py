import matplotlib
matplotlib.use('TkAgg')
import json
from matplotlib import pyplot as plt
import numpy as np
import sys

def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def main():
    env_name = sys.argv[1]
    with open(f"results/instance0/rewards_{env_name}.json") as f:
        multi_data = json.loads(f.read())
    with open(f"rewards_{env_name}_local.json") as f:
        single_data = json.loads(f.read())
    
    n = min(len(multi_data), len(single_data))
    multi_data = multi_data[:n]
    single_data = single_data[:n]

    plt.plot(moving_average(single_data, 100), label = "Single Agent")
    plt.plot(moving_average(multi_data, 100), label = "Multi Agent")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
