import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import numpy as np
import os
from matplotlib.widgets import Slider

file_names = os.listdir("/u/dem/kebl6911/Part-II/MASH_optimization/Window_data")
print(file_names)

og_exciton_data = np.loadtxt("MASH_optimization/Data/mash_exc300K.dat")
fig, ax = plt.subplots(figsize=(8, 6))

ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Select File', 0, len(file_names) - 1, valinit=0, valstep=1)


def update(val):
    file_index = int(slider.val)
    file = file_names[file_index]
    data = np.loadtxt(f"MASH_optimization/Window_data/{file}")

    ax.clear()

    # Plot the 'og_exciton_data' in the background
    for i in range(8):
        c = "C%i" % i
        ax.plot(og_exciton_data[:, 0], og_exciton_data[:, i], label=f"{i}_og", linestyle="--", color=c)

    # Plot the data from the selected file on top
    for i in range(8):
        c = "C%i" % i
        ax.plot(data[:, 0], data[:, i], label=f"{i}_ls", color=c)

    ax.set_xlabel("Time")
    ax.set_ylabel("Population(1-n)")
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.draw()

# Connect the slider to the update function
slider.on_changed(update)

# Display the initial plot
update(0)