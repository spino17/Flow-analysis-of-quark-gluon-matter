import torch
import numpy as np
import matplotlib.pyplot as plt

# ranges of the data
x = np.arange(0.01, 6.01, 0.02)
y = np.arange(10, 5010, 20)

# alternate to built the mesh
x = np.tile(x, 250).reshape(-1, 1)
y = y.reshape(-1, 1).repeat(300, 0)

xy = np.concatenate((x, y), 1)

filepath = "CorrelationData.txt"
values = []
with open(filepath) as fp:
    line = fp.readline()
    count = 1
    while line:
        if count > 1:
            z = line.split("\t")
            c = z[-1]
            digit = float(c[:-1])
            print(digit)
            values.append(int(digit))
        count += 1
        line = fp.readline()

final_values = []
start = 50
l = len(values)
while start + 50 <= l:
    end = start + 300
    final_values += values[start:end]
    start = end + 50
final_values += values[start:]
final_values = np.array(final_values)

# saving the dataset into the file
np.save("x.npy", xy)
np.save("y.npy", final_values)

# load the data from file
c = np.load("x.npy")
d = np.load("y.npy").reshape(-1, 1)

from mpl_toolkits import mplot3d

ax = plt.axes(projection="3d")
ax.plot3D(x, y, final_values, "gray")

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])


cp = ax.contour(x, y, final_values)
ax.clabel(cp, inline=True, fontsize=10)
ax.set_title("Contour Plot")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
plt.show()

plt.contour(x, y, values, colors="black", linestyles="dashed")
plt.clabel(cp, inline=True, fontsize=10)
plt.title("Contour Plot")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.show()
