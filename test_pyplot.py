import matplotlib.pyplot as plt
import numpy as np


cluters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mae = [0.74, 0.75, 0.78, 0.81, 0.83, 0.84, 0.85, 0.86, 0.86, 0.87]

x = np.linspace(0, 2, 100)

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig, ax = plt.subplots()  # Create a figure and an axes.

ax.plot(cluters, mae, 's-', label='Clustering')

ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()