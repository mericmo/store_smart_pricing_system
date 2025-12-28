import json

import requests
def get_weather():
    api_key = "ad723c66942efea536d7b3de297b80e7"  # 需注册获取
    # 有免费额度，但需要绑定信用卡支付
    url = f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat=39.099724&lon=-94.578331&date=2024-03-04&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    # https://openweathermap.org/api/one-call-3#history_daily_aggregation
    print(json.dumps(data))
get_weather()