import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import pandas as pd
import scipy.integrate as sci


def f(x):
  return np.sin(x) + 0.5 * x

a = 0.5
b = 9.5
x = np.linspace(0, 10)
y = f(x)

from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

Ix = np.linspace(a, b)
Iy = f(Ix)

verts = [(a, 0)] + list(zip(Ix, Iy)) +  [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)

plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$", horizontalalignment='center', fontsize=20)
plt.figtext(0.9, 0.075, "$x$")
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)])
plt.show()