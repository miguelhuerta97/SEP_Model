import numpy as np
from os.path import join
import re
from os import walk
from post_process.visualization import init_figure


IFOL = '/home/jp/tesis/experiments/computational_tractability'
OFOL = IFOL


def calculate_ar_times(ifol):
    _, _, l_files = next(walk(ifol))

    l_log_files = [i for i in l_files if i.startswith('log')]
    l_log_files.sort(key=lambda x: int(re.search(r'log_([0-9]+)', x).group(1)))
    l_log_files = [join(ifol, i) for i in l_log_files]

    print(l_log_files)

    n_logs = len(l_log_files)

    ar_times = np.zeros(shape=(n_logs,))

    count = 0
    for file in l_log_files:
        with open(file, 'r') as hfile:
            text = hfile.read()
            time = float(re.search(r'and ([0-9\.]+) seconds', text).group(1))
            print(time)
        ar_times[count] = time
        count += 1

    return ar_times

ar_ret = None
N = 3
for i in range(1, N + 1):
    ifol = join(IFOL, 'exp{}'.format(i))
    ar_times = calculate_ar_times(ifol)

    if ar_ret is None:
        ar_ret = np.zeros(shape=(ar_times.shape[0], N))


    ar_ret[:, i-1] = ar_times



ar_times = ar_ret.mean(axis=1)
n_logs = ar_times.shape[0]

fig, ax = init_figure(0.5 *1.1*1.1, 0.2 *1.1*1.1*1.05)

ax.plot(list(range(1, n_logs + 1)), ar_times)


ax.set_xlabel('Number of controlled DGs')
ax.set_ylabel('Computational time (s)')
ax.tick_params(direction='in')

ax.set_ylim([0, 45])
ax.set_xlim([0, 33])

fig.tight_layout()

fig.savefig(join(OFOL, 'tractability.svg'), format='svg')
fig.show()






