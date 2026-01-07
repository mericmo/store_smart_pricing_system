import os
import csv
from datetime import datetime, timedelta
import pandas as pd
from meteostat import Point, Daily

# 设置目标城市的经纬度（示例：北京 北纬39.9042，东经116.4074）
TARGET_LOCATION = Point(39.9042, 116.4074)
# CSV 文件保存路径
CSV_FILE_PATH = "weather_data.csv"

def get_and_save_weather_data():
    """
    获取天气数据并增量保存到CSV文件
    首次运行获取最近两年数据，后续运行只获取增量日期数据
    """
    # 1. 确定数据获取的起始和结束日期
    end_date = datetime.now().date()  # 结束日期为当前日期
    start_date = None

    # 检查CSV文件是否存在，判断是否需要增量更新
    if os.path.exists(CSV_FILE_PATH):
        try:
            # 读取已有数据，获取最新日期
            df = pd.read_csv(CSV_FILE_PATH)
            # 确保日期列是datetime格式
            df['time'] = pd.to_datetime(df['time']).dt.date
            # 获取已有数据的最大日期
            max_exist_date = df['time'].max()
            # 增量起始日期为已有最大日期的次日
            start_date = max_exist_date + timedelta(days=1)
            
            # 如果起始日期大于结束日期，说明数据已是最新，无需更新
            if start_date > end_date:
                print("当前天气数据已是最新，无需更新！")
                return
            print(f"检测到已有数据，将获取 {start_date} 至 {end_date} 的增量数据")
        except Exception as e:
            print(f"读取已有CSV文件失败，将重新获取最近两年数据，错误信息：{e}")
            # 重新设置起始日期为两年前
            start_date = end_date - timedelta(days=365*2)
    else:
        # 文件不存在，获取最近两年数据
        start_date = end_date - timedelta(days=365*2)
        print(f"未检测到CSV文件，将获取 {start_date} 至 {end_date} 的天气数据")

    # 2. 使用meteostat获取天气数据
    try:
        # 转换为meteostat需要的datetime对象
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # 获取每日天气数据
        data = Daily(TARGET_LOCATION, start_datetime, end_datetime)
        data = data.fetch()  # 转换为DataFrame
        
        if data.empty:
            print("未获取到任何天气数据！")
            return
        
        # 重置索引（date会从索引变为列）
        data.reset_index(inplace=True)
        # 格式化日期列（确保格式统一）
        data['time'] = data['time'].dt.date
        
        # 3. 保存/追加数据到CSV文件
        if os.path.exists(CSV_FILE_PATH):
            # 追加模式，不写入表头
            data.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
            print(f"成功追加 {len(data)} 条天气数据到 {CSV_FILE_PATH}")
        else:
            # 新建文件，写入表头
            data.to_csv(CSV_FILE_PATH, mode='w', header=True, index=False, quoting=csv.QUOTE_ALL)
            print(f"成功新建文件并写入 {len(data)} 条天气数据到 {CSV_FILE_PATH}")
            
    except Exception as e:
        print(f"获取/保存天气数据失败，错误信息：{e}")

# 测试调用
if __name__ == "__main__":
    get_and_save_weather_data()