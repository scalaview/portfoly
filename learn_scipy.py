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


>>> sci.fixed_quad(f, a, b) #固定高斯求积
(24.366995967084602, None)
>>> sci.quad(f, a, b)  #自适应求积
(24.374754718086752, 2.706141390761058e-13)
>>> sci.romberg(f, a, b)  #龙贝格积分
24.374754718086713

>>> xi = np.linspace(0.5, 9.5, 25)
>>> sci.trapz(f(xi), xi)  # 梯形法则
24.352733271544516
>>> sci.simps(f(xi), xi)    #辛普森法则
24.374964184550748

