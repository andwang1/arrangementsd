import matplotlib.pyplot as plt
import numpy as np
from exp_config import *

GEN_NUMBER = 7500

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/results_balltrajectorysd_aurora/'
DIR_PATH = 'gen8000_pctrandom0_fulllossfalse/2020-06-01_18_24_03_122344/'
FILE_NAME = f'diversity{GEN_NUMBER}.dat'

FILE = BASE_PATH + DIR_PATH + FILE_NAME

with open(FILE, "r") as f:
    lines = f.readlines()
    max_diversity = int(lines[0].strip().split(":")[-1])
    achieved_diversity = float(lines[1].strip())
    # bitmap prints in reverse order
    diversity_grid = lines[2].strip().split(",")[::-1]

rows = []
column = []
counter_x = 0
for i in diversity_grid:
    column.append(float(i))
    counter_x += 1
    if counter_x >= DISCRETISATION:
        counter_x = 0
        rows.append(column)
        column = []

# plot colours
fig = plt.figure()
plt.ylim([DISCRETISATION, 0])
plt.xlim([0, DISCRETISATION])

# vmin/vmax sets limits
color = plt.pcolormesh(rows, vmin=0, vmax=1)

# plot grid
plt.grid(which="both")
plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
fig.colorbar(color)
plt.title(f"Diversity - BinsTransversed / TotalBins - Gen {GEN_NUMBER}")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()