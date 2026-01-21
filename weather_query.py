from meteostat import Point, Daily,Monthly,Hourly
from datetime import datetime
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 北京坐标
# location = Point(39.9042, 116.4074)

#深圳罗湖
location = Point(22.56, 114.14)
# 获取2023年1月1日到1月31日的数据
start = datetime(2026, 1, 5)
end = datetime(2026, 1, 8)

data = Daily(location, start, end)
data = data.fetch()
# Index(['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres',
#        'tsun'],
# tavg	日平均气温（Average temperature）
# tmin	日最低气温（Minimum temperature）
# tmax	日最高气温（Maximum temperature）
# prcp	日降水量（Precipitation）
# snow	积雪深度（Snow depth
# wdir	平均风向（Wind direction）
# wspd	平均风速（Wind speed）
# wpgt	阵风最大风速（Peak wind gust）
# pres	平均海平面气压（Sea-level air pressure）
# tsun	实际日照时长（Sunshine duration）
#       dtype='object')
print(data)
print(data[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']])


# Get hourly data
data = Hourly('ZBTJ0', start, end)
# hourly data: Index(['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres',
#        'tsun', 'coco'],
#       dtype='object')
# rhum	相对湿度（Relative humidity）
# temp	气温（Air temperature）
# dwpt	露点温度（Dew point）
# prcp	小时降水量（Precipitation）
# snow	积雪深度（Snow depth）
# wdir	风向（Wind direction）
# wspd	风速（Wind speed）
# wpgt	阵风风速（Peak wind gust）
# pres	海平面气压（Sea-level pressure）
# tsun	日照时长（Sunshine duration）
# coco	天气状况代码（Cloud cover / Weather condition code）
data = data.fetch()
print("hourly data:",data.columns)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 6))
# 绘图
data.plot(y=['temp', 'wspd', 'prcp'],ax=ax)

# 调整布局
plt.tight_layout()

# 保存图像
fig.savefig('weather_ZBTJ0.png', dpi=300, bbox_inches='tight')

# 关闭图像释放内存
plt.close(fig)