import numpy as np, matplotlib.pyplot as plt, mplcursors

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=y, cmap='viridis')
mplcursors.cursor(sc, hover=True)
plt.show()
