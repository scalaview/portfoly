import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

bits = [ 'btccny',
  'ethcny',
  'dgdcny',
  'btscny',
  'dcscny',
  'sccny',
  'etccny',
  '1stcny',
  'repcny',
  'anscny',
  'zeccny',
  'zmccny',
  'gntcny',
  'qtumcny' ]


def normality_tests(arr):
  '''
  Tests for normality distribution of given data set.
  Parameters array: ndarray
  object to generate on
  '''
  print("Skew of data set %14.3f" % scs.skew(arr))
  print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
  print("Kurt of data set %14.3f" % scs.kurtosis(arr))
  print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
  print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])

def gen_paths(data):
  size = len(data)
  fre = 24
  paths = np.zeros((size, fre))
  i = 0
  for t in data:
    peo = []
    s_list = sorted(data[t], key=lambda k: k['time'])
    start = s_list[0]['amount']
    target = [(x['amount']/start - 1) for x in s_list]
    end = target[-1]
    if i == 0 and len(target) != fre:
      for x in range(0, fre - len(target)):
        target.insert(0, 0)
    if i == (size-1) and len(target) != fre:
      for x in range(0, fre - len(target)):
        target.append(end)
    paths[i] = np.array(target)
    i = i + 1
  return paths

def loadData(sym):
  res = requests.get("https://k.sosobtc.com/data/period?symbol=yunbi"+sym+"&step=3600")
  data = {}
  for da in json.loads(res.text):
    time = da[0]
    start = da[1]
    end = da[4]
    data_str = datetime.fromtimestamp(time).strftime('%Y-%m-%d')
    if data.get(data_str) is None:
      data[data_str] = []
    data[data_str].append({'time': time, 'amount': end})
  result = gen_paths(data)
  plt.clf()
  for x in result:
    plt.plot(x)
  plt.grid(True)
  plt.savefig("./img/"+sym+'.png', bbox_inches='tight')
  plt.clf()
  plt.hist(result, bins=30)
  plt.savefig("./img/"+sym+'-hist.png', bbox_inches='tight')
  plt.clf()
  sm.qqplot(result, line='s')
  plt.savefig("./img/"+sym+'-qq.png', bbox_inches='tight')
  normality_tests(result[-1])

if __name__ == '__main__':
  for x in bits:
    loadData(x)
