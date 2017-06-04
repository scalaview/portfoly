import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt


def gen_paths(S0, r, sigma, T, M, I):
  dt = float(T) / M
  paths = np.zeros((M+1, I), np.float64)
  paths[0] = S0
  for t in range(1, M+1):
    rand = np.random.standard_normal(I)
    rand = (rand - rand.mean()) / rand.std()
    paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
  return paths

S0 = 100
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000

paths = gen_paths(S0, r, sigma, T, M, I)

plt.plot(paths[:, :10])
plt.grid(True)
plt.show()

log_returns = np.log(paths[1:] / paths[0:-1])

'''
>>> paths[:, 0].round(4)
array([ 100.    ,   96.6164,  104.3296,  102.565 ,  102.6243,  105.0844,
        108.4835,  110.4799,  111.9269,  113.0225,  116.262 ,  111.0043,
        112.6174,  109.7845,  110.6934,  110.0979,  107.459 ,  105.0001,
        104.0568,  105.0153,  104.518 ,  108.2301,  102.8982,   99.2804,
        101.5576,   99.2155,  105.9716,  101.3538,  101.656 ,   98.532 ,
        100.6384,  103.934 ,  103.7848,  105.7799,  111.7936,  110.265 ,
        112.2557,  113.6688,  110.9802,  113.9528,  119.6442,  117.7327,
        123.4827,  117.2669,  118.1398,  121.5619,  120.4562,  119.7591,
        115.6617,  111.47  ,  104.1384])
>>> log_returns[:, 0].round(4)
array([-0.0344,  0.0768, -0.0171,  0.0006,  0.0237,  0.0318,  0.0182,
        0.013 ,  0.0097,  0.0283, -0.0463,  0.0144, -0.0255,  0.0082,
       -0.0054, -0.0243, -0.0231, -0.009 ,  0.0092, -0.0047,  0.0349,
       -0.0505, -0.0358,  0.0227, -0.0233,  0.0659, -0.0446,  0.003 ,
       -0.0312,  0.0212,  0.0322, -0.0014,  0.019 ,  0.0553, -0.0138,
        0.0179,  0.0125, -0.0239,  0.0264,  0.0487, -0.0161,  0.0477,
       -0.0516,  0.0074,  0.0286, -0.0091, -0.0058, -0.0348, -0.0369,
       -0.068 ])
'''

# 输出数组特征
def print_statistics(array):
    sta = scs.describe(array)
    print("%14a %15s" % ('statistic', 'value'))
    print(30 * ".")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std' , np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew' , sta[4]))
    print("%14s %15.5f" % ('kurtosis' , sta[5]))

print_statistics(log_returns.flatten())
'''
   'statistic'           value
..............................
          size  12500000.00000
           min        -0.15438
           max         0.14645
          mean         0.00060
           std         0.02828
          skew         0.00067
      kurtosis         0.00088
'''

plt.hist(log_returns.flatten())
