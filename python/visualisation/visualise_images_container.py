import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from exp_config import *

# save every nth image
nth = 100

FULL_PATH = sys.argv[1]
os.chdir(FULL_PATH)
files = os.listdir()
image_dat_files = []

# Find generation numbers
img_generations = []
for fname in files:
    if fname.startswith("images_") and ".dat" in fname:
        img_generations.append(fname.rstrip(r".dat")[len("images_"):])
        image_dat_files.append(fname)

dist_generations = sorted(int(gen) for gen in img_generations)
GEN_NUMBER = dist_generations[-1]

FILE_NAME = f'images_{GEN_NUMBER}.dat'

FILE = FULL_PATH + "/" + FILE_NAME

with open(FILE, 'r') as f:
    lines = f.readlines()[1:]

# list of lists of recon, trajectories, losses
plotting_data = []

num_individuals = int(lines[-1].strip().split(",")[0])

line_number = 0
file_length = len(lines)

# collect the data from the file
while line_number < file_length:
    if int(lines[line_number].strip().split(",")[0]) % nth != 0:
        line_number += 1
        continue

    indiv_data = []
    indiv_index = int(lines[line_number].strip().split(",")[0])
    while line_number != file_length:
        data = lines[line_number].strip().split(",")
        line_number += 1
        if int(data[0]) != indiv_index:
            data.append(indiv_data)
            break
        indiv_data.append([float(i) for i in data[2:]])
    plotting_data.append(indiv_data)

# plot
if "vae" in FILE:
    for j, indiv in enumerate(plotting_data):
        f = plt.figure(figsize=(30, 20))
        spec = f.add_gridspec(3, 2)
        # both kwargs together make the box squared
        ax1 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')
        prediction = indiv[0]
        prediction = np.array(prediction).reshape((DISCRETISATION, DISCRETISATION))
        ax1.imshow(prediction, vmin=0, vmax=1, cmap="gray")

        ax1.set_ylim([0, ROOM_H])
        ax1.set_xlim([0, ROOM_W])

        actual = indiv[1]
        ax2 = f.add_subplot(spec[0, 1], aspect='equal', adjustable='box')

        actual = np.array(actual).reshape((DISCRETISATION, DISCRETISATION))
        ax2.imshow(actual, vmin=0, vmax=1, cmap="gray")

        ax2.set_ylim([0, ROOM_H])
        ax2.set_xlim([0, ROOM_W])
        ax2.legend(loc="best")

        noise_free_actual = indiv[2]
        ax3 = f.add_subplot(spec[0, 2], aspect='equal', adjustable='box')

        noise_free_actual = np.array(noise_free_actual).reshape((DISCRETISATION, DISCRETISATION))
        ax3.imshow(noise_free_actual, vmin=0, vmax=1, cmap="gray")

        ax3.set_ylim([0, ROOM_H])
        ax3.set_xlim([0, ROOM_W])

        ax1.set_title("Construction")
        ax2.set_title("Actual Observation")
        ax3.set_title("Noise-Free Observation")

        L2 = np.array(indiv[3]).reshape(DISCRETISATION, DISCRETISATION)
        ax3 = f.add_subplot(spec[1, 0], aspect='equal', adjustable='box')
        divider1 = make_axes_locatable(ax3)
        cax1 = divider1.append_axes("left", size="5%", pad=0.45)
        cax1.yaxis.set_ticks_position('left')
        cax1.yaxis.set_label_position('left')
        sns.heatmap(L2, ax=ax3, vmin=0, cbar_ax=cax1,
                    xticklabels=np.arange(0, ROOM_W, ROOM_W / DISCRETISATION),
                    yticklabels=np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
        cax1.yaxis.set_ticks_position('left')
        cax1.yaxis.set_label_position('left')
        ax3.set_title("L2")
        ax3.invert_yaxis()

        var = np.array(indiv[-1]).reshape(DISCRETISATION, DISCRETISATION)
        ax4 = f.add_subplot(spec[1, 1], aspect='equal', adjustable='box')
        divider2 = make_axes_locatable(ax4)
        cax2 = divider2.append_axes("right", size="5%", pad=0.45)
        sns.heatmap(var, ax=ax4, vmin=0, cbar_ax=cax2,
                    xticklabels=np.arange(0, ROOM_W, ROOM_W / DISCRETISATION),
                    yticklabels=np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
        ax4.set_title("Variance")
        ax4.invert_yaxis()

        plt.subplots_adjust(hspace=0.6)
        plt.savefig(f"image_{j}.png")
        plt.close()

else:
    for j, indiv in enumerate(plotting_data):
        f = plt.figure(figsize=(20, 20))
        spec = f.add_gridspec(2, 2)
        # both kwargs together make the box squared
        ax1 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')
        prediction = indiv[0]
        prediction = np.array(prediction).reshape((DISCRETISATION, DISCRETISATION))
        ax1.imshow(prediction, vmin=0, vmax=1, cmap="gray")

        ax1.set_ylim([0, ROOM_H])
        ax1.set_xlim([0, ROOM_W])

        actual = indiv[1]
        ax2 = f.add_subplot(spec[0, 1], aspect='equal', adjustable='box')

        actual = np.array(actual).reshape((DISCRETISATION, DISCRETISATION))
        ax2.imshow(actual, vmin=0, vmax=1, cmap="gray")

        ax2.set_ylim([0, ROOM_H])
        ax2.set_xlim([0, ROOM_W])
        ax2.legend(loc="best")

        noise_free_actual = indiv[2]
        ax3 = f.add_subplot(spec[1, 0], aspect='equal', adjustable='box')

        noise_free_actual = np.array(noise_free_actual).reshape((DISCRETISATION, DISCRETISATION))
        ax3.imshow(noise_free_actual, vmin=0, vmax=1, cmap="gray")

        ax3.set_ylim([0, ROOM_H])
        ax3.set_xlim([0, ROOM_W])

        ax1.set_title("Construction")
        ax2.set_title("Actual Observation")
        ax3.set_title("Noise-Free Observation")

        rows = []
        column = []
        counter_x = 0
        for i in indiv[2]:
            column.append(float(i))
            counter_x += 1
            if counter_x >= DISCRETISATION:
                counter_x = 0
                rows.append(column)
                column = []

        L2 = np.array(indiv[3]).reshape(DISCRETISATION, DISCRETISATION)
        ax3 = f.add_subplot(spec[1, 1], aspect='equal', adjustable='box')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        sns.heatmap(L2, ax=ax3, vmin=0, vmax=1, cbar_ax=cax,
                    xticklabels=np.arange(0, ROOM_W, ROOM_W / DISCRETISATION),
                    yticklabels=np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
        ax3.invert_yaxis()
        ax3.set_title("L2")

        plt.subplots_adjust(hspace=0.6)
        plt.savefig(f"image_{j}.png")
        plt.close()

os.makedirs("images", exist_ok=True)
image_files = [img for img in os.listdir() if ".png" in img]
for image in image_files:
    shutil.move(image, f"images/{image}")

for file in image_dat_files:
    os.remove(file)