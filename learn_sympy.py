import sympy as sy
import scipy.stats as scs


x = sy.Symbol('x')
y = sy.Symbol('y')
type(x)
<class 'sympy.core.symbol.Symbol'>
sy.sqrt(x)
sqrt(x)
3 + sy.sqrt(x) - 4**2
sqrt(x) - 13
f = x ** 2 + 3 + 0.5*x **2 + 3/2
>>>
sy.simplify(f)
1.5*x**2 + 4.5
sy.init_printing(pretty_print=False, use_unicode=False)
print(sy.pretty(f))
     2
1.5*x  + 4.5
print(sy.pretty(sy.sqrt(x) + 0.5))
  ___
\/ x  + 0.5
# 解方程
sy.solve(x ** 2 - 1)
[-1, 1]
sy.solve(x ** 3 + 0.5 * x ** 2 - 1)
[0.858094329496553, -0.679047164748276 - 0.839206763026694*I, -0.679047164748276 + 0.839206763026694*I]
sy.solve(x ** 2 + y ** 2)
[{x: -I*y}, {x: I*y}]

# 积分
a, b = sy.symbols('a b')
print(sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))))
  b
  /
 |
 |  (0.5*x + sin(x)) dx
 |
/
a
# 积分
# 求出反导
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print(sy.pretty(int_func))
      2
0.25*x  - cos(x)
Fb = int_func.subs(x, 0.95).evalf()
Fa = int_func.subs(x, 0.5).evalf()
Fb = int_func.subs(x, 9.5).evalf()
Fb - Fa
24.3747547180867

# 求出双界的反导公式
int_func_limts = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
print(sy.pretty(int_func_limts))
        2         2
- 0.25*a  + 0.25*b  + cos(a) - cos(b)
int_func_limts.subs({a: 0.5, b: 9.5}).evalf()
24.3747547180868
# 直接计算积分
sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5))
24.3747547180867



f = (sy.sin(x) + 0.05 * x ** 2 + sy.sin(y) + 0.05 * y ** 2)
# 对x求偏导
del_x = sy.diff(f, x)
del_x
0.1*x + cos(x)

# 对y求偏导
del_y = sy.diff(f, y)
del_y
0.1*y + cos(y)

'''  求解方程，得不到解，因此需要使用近似解
>>> sy.solve(del_x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/zbin/anaconda/lib/python3.6/site-packages/sympy/solvers/solvers.py", line 1053, in solve
    solution = _solve(f[0], *symbols, **flags)
  File "/Users/zbin/anaconda/lib/python3.6/site-packages/sympy/solvers/solvers.py", line 1619, in _solve
    raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))
NotImplementedError: multiple generators [x, cos(x)]
No algorithms are implemented to solve equation x/10 + cos(x)
'''
# 求得近似解，后面的参数表示从某个数字（x）开始得到第一近似解
xo = sy.nsolve(del_x, -1.5)
xo
-1.42755177876459

yo = sy.nsolve(del_y, -1.5)
yo
-1.42755177876459


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

sample_size = 500
# 均值为0，标准差为1的标准正太分布
rn1 = npr.standard_normal(sample_size)
# 均值为100， 标准差为20的正太分布
rn2 = npr.normal(100, 20, sample_size)
# 自由度为0.5的卡方分布
rn3 = npr.chisquare(df=0.5 size=sample_size)
# 入值为1的柏松分布
rn4 = npr.poisson(lam=1.0, size=sample_size)

# Black-Scholes-Merton模型

S0 = 100
r = 0.05
sigma = 0.25
T = 2.0
I = 10000
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))
plt.hist(ST1, bins=50)
'''
(array([  1.10000000e+01,   5.30000000e+01,   2.01000000e+02,
         4.43000000e+02,   6.72000000e+02,   8.68000000e+02,
         9.86000000e+02,   1.04900000e+03,   9.92000000e+02,
         9.04000000e+02,   7.38000000e+02,   6.80000000e+02,
         5.28000000e+02,   4.27000000e+02,   3.32000000e+02,
         2.50000000e+02,   2.08000000e+02,   1.73000000e+02,
         1.17000000e+02,   9.00000000e+01,   6.70000000e+01,
         4.60000000e+01,   4.30000000e+01,   3.30000000e+01,
         2.50000000e+01,   1.60000000e+01,   8.00000000e+00,
         1.00000000e+01,   5.00000000e+00,   6.00000000e+00,
         4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
         2.00000000e+00,   1.00000000e+00,   1.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
         1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.00000000e+00]),
array([  26.20663435,   35.08198541,   43.95733647,   52.83268753,
         61.70803859,   70.58338965,   79.45874071,   88.33409177,
         97.20944283,  106.08479389,  114.96014495,  123.83549601,
        132.71084707,  141.58619813,  150.46154919,  159.33690025,
        168.21225131,  177.08760237,  185.96295343,  194.83830449,
        203.71365555,  212.58900661,  221.46435767,  230.33970873,
        239.21505979,  248.09041085,  256.96576191,  265.84111297,
        274.71646403,  283.59181509,  292.46716615,  301.34251721,
        310.21786827,  319.09321933,  327.96857039,  336.84392145,
        345.71927251,  354.59462357,  363.46997463,  372.34532569,
        381.22067675,  390.09602781,  398.97137887,  407.84672993,
        416.72208099,  425.59743205,  434.47278311,  443.34813417,
        452.22348523,  461.09883629,  469.97418735]), <a list of 50 Patch objects>)
'''
plt.grid(True)
plt.show()


# 根据ST1随机变量符合对数正态分布，使用lognormal进行模拟
S2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T), size=I)
plt.hist(S2, bins=50)
'''
(array([   8.,   52.,  127.,  339.,  529.,  755.,  923.,  963.,  939.,
        941.,  770.,  692.,  637.,  498.,  393.,  307.,  237.,  200.,
        126.,  128.,  103.,   62.,   63.,   40.,   28.,   41.,   25.,
         18.,   12.,    6.,    6.,    3.,    3.,    5.,    4.,    2.,
          2.,    2.,    5.,    1.,    1.,    0.,    1.,    0.,    1.,
          1.,    0.,    0.,    0.,    1.]), array([  25.42617276,   33.75013302,   42.07409328,   50.39805354,
         58.7220138 ,   67.04597406,   75.36993432,   83.69389458,
         92.01785484,  100.3418151 ,  108.66577536,  116.98973562,
        125.31369588,  133.63765614,  141.9616164 ,  150.28557666,
        158.60953692,  166.93349718,  175.25745744,  183.5814177 ,
        191.90537796,  200.22933822,  208.55329848,  216.87725874,
        225.201219  ,  233.52517926,  241.84913952,  250.17309977,
        258.49706003,  266.82102029,  275.14498055,  283.46894081,
        291.79290107,  300.11686133,  308.44082159,  316.76478185,
        325.08874211,  333.41270237,  341.73666263,  350.06062289,
        358.38458315,  366.70854341,  375.03250367,  383.35646393,
        391.68042419,  400.00438445,  408.32834471,  416.65230497,
        424.97626523,  433.30022549,  441.62418575]), <a list of 50 Patch objects>)
'''
plt.grid(True)
plt.show()

# 使用下列函数对比ST1于S2的相似度
def print_statistics(a1, a2):
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print("%14a %14a %14s" % ('statistic', 'data set 1' , 'dataset 2'))
    print(45 * ".")
    print("%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0]))
    print("%14s %14.3f %14.3f" % ('min', sta1[1][0], sta2[1][0]))
    print("%14s %14.3f %14.3f" % ('max', sta1[1][1], sta2[1][1]))
    print("%14s %14.3f %14.3f" % ('mean', sta1[2], sta2[2]))
    print("%14s %14.3f %14.3f" % ('std' , np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print("%14s %14.3f %14.3f" % ('skew' , sta1[4], sta2[4]))
    print("%14s %14.3f %14.3f" % ('kurtosis' , sta1[5], sta2[5]))

print_statistics(ST1, S2)
'''
   'statistic'   'data set 1'      dataset 2
.............................................
          size      10000.000      10000.000
           min         26.207         25.426
           max        469.974        441.624
          mean        110.391        110.506
           std         40.291         41.063
          skew          1.199          1.332
      kurtosis          2.871          3.508
'''


# 几何布朗运动模拟
I = 10000
M = 50
dt = T/M
S = np.zeros((M+1, I))
S[0] = S0
for t in range(1, M+1):
  S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*npr.standard_normal(I))


plt.hist(S[-1], bins=50)
plt.grid(True)
plt.show()

# 10条模拟路径
plt.plot(S[:, :10], lw=1.5)
plt.grid(True)
plt.show()

# 平方根扩散
x0 = 0.05
kappa = 3.0
theta = 0.02
sigma = 0.1

I = 10000
M = 50
dt = I / 50

def srd_euler():
  xh = np.zeros((M+1, I))
  x1 = np.zeros_like(xh)
  xh[0] = x0
  x1[0] = x0
  for t in range(1, M+1):
    xh[t] = (xh[t-1] + kappa * (theta - np.maximum(xh[t-1], 0)) * dt \
    + sigma * np.sqrt(np.maximum(xh[t-1], 0)) * np.sqrt(dt) * npr.standard_normal(I))
  x1 = np.maximum(xh, 0)
  return x1
x1 = srd_euler()

plt.hist(x1[-1], bins=50)
plt.grid(True)
plt.show()

plt.plot(x1[:, :10], lw=1.5)
plt.grid(True)
plt.show()



S0 = 100
r = 0.05
v0 = 0.1
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6
T = 1.0


corr_mat = np.zeros((2, 2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat)

M = 50
I = 10000
ran_num = npr.standard_normal((2, M + 1, I))

dt = T / M
v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)
v[0] = v0
vh[0] = v0
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :])
  vh[t] = (vh[t-1] + kappa * (theta - np.maximum(vh[t-1], 0)) * dt + sigma * np.sqrt(np.maximum(vh[t-1], 0)) * np.sqrt(dt) + ran[1])

v = np.maximum(vh, 0)

S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, M+1):
  ran = np.dot(cho_mat, ran_num[:, t, :])
  S[t] = S[t-1] * np.exp((r-0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
ax1.hist(S[-1], bins=50)
ax1.grid(True)

ax2.hist(v[-1], bins=50)
ax2.grid(True)














