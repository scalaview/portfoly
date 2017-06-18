import warning
warning.simplefilter('ignore')
import zipline
import pytz
import datetime as dt

data = zipline.data.load_from_yahoo(stocks=['GLD', 'GDX'], end=dt.datetime(2014, 3, 15, 0, 0, 0, 0, pytz.utc)).dropna()
data.info()


import bokeh.plotting as bp
# 图片会被输出称为一个html文件
bp.output_file("../images/msft_l.html", title="Bokeh Example (Stati c)")