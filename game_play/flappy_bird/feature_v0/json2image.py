import json
import numpy as np
from matplotlib import pyplot as plt

with open('run-log_DQN_1-tag-eval_mean_ep_length.json', 'r') as f:
    str = f.read()

log_list = json.loads(str)

x = [log[1] for log in log_list]
y = [log[2] for log in log_list]

x = np.array(x)
y = np.array(y)
plt.plot(x, y)

if __name__ == '__main__':
    plt.show()