
from models.prophet_model import ProphetModel
from data.daily_preprocessor import DailyPreprocessor
import pandas as pd
from utils import common, calender_helper, save_to_csv
from data.daily_feature_store import DailyFeatureStore
product_code = '4701098'
store_code = '205625'

def train_prophet():

    # features_df = pd.read_csv('temp/5_preprocess_sales_data.csv', encoding='utf-8')
    features_df = pd.read_csv('data/historical_transactions.csv', encoding='utf-8', parse_dates=['日期', '交易时间'],  dtype={ "商品编码": str, "门店编码": str, "商品小类": str})


    date_series = features_df['日期'].unique()
    calendar_df = calender_helper.create_china_holidays_from_date_list(date_series=date_series)

    weather_df = pd.read_csv('data/weather_info.csv', encoding='utf-8')


    # 1. 基础预处理
    data_preprocessor = DailyPreprocessor(product_code, store_code)
    sales_processed = data_preprocessor.preprocess_sales_data(features_df)
    weather_processed = data_preprocessor.preprocess_weather_data(weather_df)
    calendar_processed = data_preprocessor.preprocess_calendar_data(calendar_df)

    # 初始化特征仓库
    feature_store = DailyFeatureStore()
    # 2. 时间特征
    sales_with_time = feature_store.create_time_features(sales_processed)

    # 3. 滞后特征（需要按商品单独计算）
    sales_with_lag = feature_store.create_lag_features(sales_with_time)

    # 4. 滚动特征（需要按商品单独计算）
    sales_with_rolling = feature_store.create_rolling_features(sales_with_lag)

    # 5. 趋势特征
    sales_with_trend = feature_store.create_trend_features(sales_with_rolling)

    # 6. 合并天气特征
    sales_with_weather = feature_store.create_weather_enhanced_features(
        sales_with_trend, weather_processed
    )

    # 7. 合并日历特征
    sales_with_calendar = feature_store.create_calendar_enhanced_features(
        sales_with_weather, calendar_processed
    )
    save_to_csv(sales_with_calendar)
    # 8. 交互特征
    final_features = feature_store.create_interaction_features(sales_with_calendar)

    save_to_csv(final_features)
    # 初始化预测器
    predictor = ProphetModel()

    # 1. 单独训练Prophet模型
    _, ph_data = predictor.train_prophet(
        final_features,
        target_col='销售数量'
    )

    # 3. 使用Prophet进行预测
    result = predictor.predict_prophet(
        final_features,
        periods=1,  # 预测未来30天
        freq='D'     # 按天预测 D
    )
    print("预测结果",result)
    # 4. 绘制Prophet预测结果
    fig = predictor.plot_prophet_forecast(final_features, forecast_periods=30)
    fig.show()

    # 5. 绘制Prophet模型组件
    components_fig = predictor.plot_prophet_components(final_features)
    components_fig.show()

    # 6. 获取Prophet评估结果
    prophet_eval = predictor.get_evaluation_results('prophet')
    print(f"Prophet模型: {prophet_eval}")

if __name__ == '__main__':
    train_prophet()