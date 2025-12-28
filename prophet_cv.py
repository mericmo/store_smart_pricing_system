
from models.prophet_model import ProphetModel
import pandas as pd
# 初始化预测器
predictor = ProphetModel()
features_df = pd.read_csv('temp/6_preprocess_sales_data.csv', encoding='utf-8')
# 1. 单独训练Prophet模型
prophet_model = predictor.train_prophet(
    features_df,
    target_col='销售数量'
)

# 2. 训练所有模型（包括Prophet）
all_models = predictor.train_prophet(features_df)

# 3. 使用Prophet进行预测
forecast_result = predictor.predict_prophet(
    features_df,
    periods=30,  # 预测未来30天
    freq='D'     # 按天预测 D
)

# 4. 绘制Prophet预测结果
fig = predictor.plot_prophet_forecast(features_df, forecast_periods=30)
fig.show()

# 5. 绘制Prophet模型组件
components_fig = predictor.plot_prophet_components(features_df)
components_fig.show()

# 6. 获取Prophet评估结果
prophet_eval = predictor.get_evaluation_results('prophet')
print(f"Prophet模型RMSE: {prophet_eval['RMSE']:.2f}")