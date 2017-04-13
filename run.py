import requests
import json

class Subject():
    def __init__(self, d):
        self.__dict__ = d


class Origin(object):

  timeout = 100

  __urls = {
    "rank": "http://103.37.160.30:12150/v1/portfolios/leaderboards2.json?app=quant&client=iPhone%205C&version=1.6.0.3&fee=paid&orderby=returns",
  }

  def __init__(self):
    self.init_opener()

  def init_opener(self, headers = {
    'Connection': 'Keep-Alive',
    'Accept': 'application/json,*/*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Appcelerator Titanium/5.5.1 (iPhone/9.3.5; iPhone OS; zh_CN;)',
    }):
    self.headers = headers
    response = requests.get(self.__urls["rank"], headers=headers, timeout=self.timeout)
    if response.status_code == 200:
      self.cookies = response.cookies
    else:
      response.raise_for_status()

class App(Origin):

    __urls = {
        "rank": "http://103.37.160.30:12150/v1/portfolios/leaderboards2.json?app=quant&client=iPhone%205C&version=1.6.0.3&fee=paid&orderby=returns",
    }

    """docstring for App"""
    def __init__(self):
        super(App, self).__init__()

    def run(self):
        res = requests.get(self.__urls["rank"], headers=self.headers, cookies=self.cookies, timeout=self.timeout)
        if res.status_code == 200:
            result = json.loads(res.text)
            for x in result[:10]:
                print(x)
        else:
            res.raise_for_status()


if  __name__ == '__main__':
    app = App()
    app.run()
