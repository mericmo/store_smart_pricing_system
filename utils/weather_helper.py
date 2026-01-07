# import json
#
# import requests
# def get_weather():
#     api_key = "ad723c66942efea536d7b3de297b80e7"  # 需注册获取
#     # 有免费额度，但需要绑定信用卡支付
#     url = f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat=39.099724&lon=-94.578331&date=2024-03-04&appid={api_key}"
#     response = requests.get(url)
#     data = response.json()
#     # https://openweathermap.org/api/one-call-3#history_daily_aggregation
#     print(json.dumps(data))
# get_weather()

import urllib, urllib3, sys, uuid
import ssl


host = 'https://ali-weather.showapi.com'
path = '/weatherhistory'
method = 'GET'
appcode = '4e48a327e1504f5c95c670389e0634e4'
querys = 'areaCode=&area=深圳&month=&startDate=20240101&endDate=20260101'
bodys = {}
url = host + path + '?' + querys

http = urllib3.PoolManager()
headers = {
    'Authorization': 'APPCODE ' + appcode
}
response = http.request('GET', url, headers=headers)
content = response.data.decode('utf-8')
if (content):
    print(content)
