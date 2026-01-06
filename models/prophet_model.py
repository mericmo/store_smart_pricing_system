from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.common import save_to_csv
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pio.templates.default = "plotly_white"


class ProphetModel:
    # 在现有代码基础上添加以下方法
    def __init__(self, params=None):
        self.models = {}
        self.evaluation_results = {}
        self.params = params or {
                'seasonality_mode': 'multiplicative',  # 乘法季节性
                'yearly_seasonality': True,  # 年季节性
                'weekly_seasonality': True,  # 周季节性
                'daily_seasonality': False,  # 日季节性（数据按天）
                'holidays_prior_scale': 10,  # 节假日影响强度
                'seasonality_prior_scale': 10,  # 季节性影响强度
                'changepoint_prior_scale': 0.05,  # 趋势变化点灵敏度
                'n_changepoints': 25,  # 变化点数量
                'interval_width': 0.95,  # 置信区间宽度
                'mcmc_samples': 0,  # 关闭MCMC采样以加速
            }
        self.scalers = {}
        self.scalers_columns = [
            "年份", '温度', '最高温度', '最低温度', '温度差',
        ]
        self.label_encoders = {}
    def train_prophet(self, features_df, target_col='销售数量', params=None):
        """
        训练Prophet模型
        Prophet是Facebook专门为时间序列预测设计的模型，适合处理节假日和季节性

        参数:
        - features_df: 特征数据框
        - target_col: 目标列名
        - params: Prophet模型参数

        返回:
        - Prophet模型
        """
        if params is None:
            params = self.params

        print("开始训练Prophet模型...")

        # 由于Prophet需要特定的数据格式，我们需要重新准备数据
        # Prophet要求两列：ds (datetime类型) 和 y (数值类型)

        # 准备Prophet数据
        prophet_data = self._prepare_prophet_data(features_df, target_col)

        if prophet_data is None or len(prophet_data) < 14:  # 至少需要2周数据
            print(f"数据不足或格式不正确，无法训练Prophet模型")
            return None
        data_length_days = (prophet_data['ds'].max() - prophet_data['ds'].min()).days
        # 计算合适的初始窗口大小以避免季节性警告
        if data_length_days < 365:  # 如果数据不足一年
            # 动态调整参数以适应较短的数据
            adjusted_params = self.params.copy()
            adjusted_params['yearly_seasonality'] = False  # 禁用年季节性
            adjusted_params['changepoint_prior_scale'] = min(0.05, self.params.get('changepoint_prior_scale', 0.05))

            # 动态设置初始窗口大小
            initial_window = max(30, data_length_days // 4)  # 至少30天
        else:
            # 数据超过一年，使用原始参数
            adjusted_params = self.params.copy()
            initial_window = min(365, max(180, data_length_days // 3))

        # 创建Prophet模型
        model = Prophet(**adjusted_params)

        # 动态设置初始窗口大小
        model.initial = initial_window

        # 添加中国的节假日
        self._add_chinese_holidays(model)
        # 数据清晰
        numeric_features = self._get_numeric_features(features_df)
        categorical_features = self._get_categorical_features(features_df)
        features_columns = [col for col in numeric_features + categorical_features
                            if col not in ['门店编码', '商品编码', '商品名称', '日期', '销售金额', '折扣金额',
                                           target_col]]
        features_df = features_df[features_columns].copy()

        # 添加额外回归因子
        self._add_regressors(model, features_df)

        # 训练模型
        prophet_data = prophet_data[:-1]
        model.fit(prophet_data)

        # 保存模型
        self.models['prophet'] = model

        # 进行交叉验证评估
        self._evaluate_prophet_model(model, prophet_data)

        print("Prophet模型训练完成!")

        return model, prophet_data
    def _get_numeric_features(self, df):
        """获取数值型特征"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in ['商品编码']]

    def _get_categorical_features(self, df):
        """获取分类特征"""
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        return [col for col in categorical_cols if col not in ['商品名称', '商品编码', '日期']]#'门店编码', '商品编码',

    def _prepare_prophet_data(self, features_df, target_col='销售数量'):
        """
        准备Prophet模型所需的数据格式

        Args:
            features_df: 原始特征数据
            target_col: 目标列名

        Returns:
            符合Prophet格式的数据框
        """
        try:
            # 复制数据以避免修改原始数据
            df = features_df.copy()

            # 确保有日期列
            if '日期' not in df.columns:
                print("错误: 数据中缺少'日期'列")
                return None

            # 确保日期列是datetime类型
            # df['日期'] = pd.to_datetime(df['日期'])

            # 按日期聚合数据（因为Prophet处理的是时间序列，通常按天）
            # 如果需要按门店和商品分别预测，这里需要调整
            # if '门店编码' in df.columns and '商品编码' in df.columns:
            #     # 按日期、门店、商品分组
            #     agg_data = df.groupby(['日期', '门店编码', '商品编码'])[target_col].sum().reset_index()
            #     # 这里简化处理，取所有门店和商品的总和，实际应用中可能需要分别建模
            #     # 或者使用分层预测
            #     prophet_data = df.groupby('日期')[target_col].sum().reset_index()
            # else:
            #     prophet_data = df.groupby('日期')[target_col].sum().reset_index()
            # df['日期时间'] = pd.to_datetime(df['日期']) + pd.to_timedelta(df['小时'], unit='h')
            # prophet_data = df.groupby('日期时间')[target_col].sum().reset_index()
            # 重命名列以符合Prophet格式
            prophet_data = df.rename(columns={
                '日期': 'ds',
                target_col: 'y'
            })

            # 确保y没有负值（如果有的话，Prophet可以处理但建议调整）
            prophet_data['y'] = prophet_data['y'].clip(lower=0)

            # 按日期排序
            prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
            save_to_csv(prophet_data)
            print(f"Prophet数据准备完成，共{len(prophet_data)}天数据")
            print(f"日期范围: {prophet_data['ds'].min()} 到 {prophet_data['ds'].max()}")

            return prophet_data

        except Exception as e:
            print(f"准备Prophet数据时出错: {e}")
            return None

    def _add_chinese_holidays(self, model):
        """添加中国节假日"""
        # 根据数据的时间范围动态生成节假日
        # 获取数据的年份范围
        if hasattr(self, '_data_min_date') and hasattr(self, '_data_max_date'):
            min_year = self._data_min_date.year
            max_year = self._data_max_date.year
        else:
            # 默认使用近3年的节假日
            current_year = pd.Timestamp.now().year
            min_year = current_year - 2
            max_year = current_year + 1

        # 生成对应年份的节假日
        holidays_list = []

        for year in range(min_year, max_year + 1):
            # 添加固定节假日
            holidays_list.extend([
                f"{year}-01-01",  # 元旦
                f"{year}-05-01",  # 劳动节
                f"{year}-10-01",  # 国庆节
                f"{year}-10-02",  # 国庆节
                f"{year}-10-03",  # 国庆节
                f"{year}-10-04",  # 国庆节
                f"{year}-10-05",  # 国庆节
                f"{year}-10-06",  # 国庆节
                f"{year}-10-07",  # 国庆节
            ])

            # 添加春节（需要根据农历转换，这里简化处理）
            if year == 2023:
                holidays_list.extend(["2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27"])
            elif year == 2024:
                holidays_list.extend(["2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16"])
            elif year == 2025:
                holidays_list.extend(["2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04"])

        # 添加其他节假日
        for year in range(min_year, max_year + 1):
            holidays_list.extend([
                f"{year}-04-04",  # 清明节
                f"{year}-04-05",
                f"{year}-04-06",
                f"{year}-06-10",  # 端午节（2024年）
                f"{year}-09-15",  # 中秋节（2024年）
                f"{year}-09-16",
                f"{year}-09-17",
            ])

        if len(holidays_list) > 0:
            chinese_holidays = pd.DataFrame({
                'holiday': 'chinese_holiday',
                'ds': pd.to_datetime(holidays_list),
                'lower_window': -2,  # 节前2天
                'upper_window': 2,   # 节后2天
            })

            # 添加节假日到模型
            if hasattr(model, 'holidays') and model.holidays is not None:
                model.holidays = pd.concat([model.holidays, chinese_holidays], ignore_index=True)
            else:
                model.holidays = chinese_holidays
    def encode_categorical_features(self,df,categorical_features):
        # 处理分类特征 - 对每个分类特征进行编码
        for col in categorical_features:
            if col in df.columns:
                # 处理可能的未知值
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # 处理可能的未知值
                    df[col] = df[col].astype(str)
                    self.label_encoders[col].fit(df[col])
                df[col] = self.label_encoders[col].transform(df[col])
        save_to_csv(df)
        return df

    def standard_scaler_features(self, features_df):
        # 限制数值范围，避免过大值
        X = features_df.copy()
        for col in self.scalers_columns:
            if col in X.columns:
                max_val = X[col].quantile(0.99)
                min_val = X[col].quantile(0.01)
                X[col] = X[col].clip(min_val, max_val)

        # 标准化数值特征（对某些算法有帮助）
        if 'standard_scaler' not in self.scalers:
            self.scalers['standard_scaler'] = StandardScaler()
            X_scaled = self.scalers['standard_scaler'].fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_scaled = self.scalers['standard_scaler'].transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        save_to_csv(X_scaled)
        return X_scaled
    def _add_regressors(self, model, features_df):
        """
        添加额外回归因子到Prophet模型

        Args:
            model: Prophet模型实例
            features_df: 特征数据框
        """
        # 选择可能对销售有影响的额外回归因子
        regressor_cols = []

        # 天气相关特征
        weather_cols = ['温度', '最高温度', '最低温度', '温度差', '降雨量', '天气严重程度']
        for col in weather_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        # 时间相关特征
        time_cols = ['是否周末', '是否月末', '是否节假日', '节假日前一天', '节假日后一天', '节假日连续天数', '月份', '星期几', '季度',  '年份']#'季节',
        for col in time_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        # 促销相关特征:'促销次数'
        promo_cols = ['是否促销', '平均售价', '实际折扣率', '销量_滞后1天', '销量_滞后3天', '销量_滞后7天', '销量_滞后14天', '销量_滞后30天', '销量_7天均值', '销量_7天标准差', '销量_14天均值', '销量_14天标准差', '销量_30天均值', '销量_30天标准差', '销量_7天趋势', '销量_同比上周', '促销_周末交互', '促销_月末交互', '天气_促销交互']
        for col in promo_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        features_df = features_df[regressor_cols].copy()
        # numeric_features = self._get_numeric_features(features_df)
        categorical_features = self._get_categorical_features(features_df)
        # features_df = self.encode_categorical_features(features_df, categorical_features)
        #
        # features_df = self.standard_scaler_features(features_df)
        # 添加回归因子
        for col in regressor_cols:
            try:
                # 按日期聚合回归因子（取平均值）
                if col in features_df.columns and len(features_df[col].unique()) > 1:
                    # regressor_data = features_df.groupby('日期')[col].mean().reset_index()
                    # regressor_data = regressor_data.rename(columns={'日期': 'ds', col: col})

                    # 添加回归因子到模型
                    model.add_regressor(col)
                    print(f"已添加回归因子: {col}")
            except Exception as e:
                print(f"添加回归因子 {col} 时出错: {e}")

    def _evaluate_prophet_model(self, model, prophet_data, cv_horizon='7 days'):
        """
        评估Prophet模型性能

        Args:
            model: Prophet模型
            prophet_data: 训练数据
            cv_horizon: 交叉验证的时间跨度
        """
        try:
            from prophet.diagnostics import cross_validation, performance_metrics

            # 进行时间序列交叉验证
            df_cv = cross_validation(
                model,
                horizon=cv_horizon,  # 预测未来30天
                period='15 days',  # 每15天进行一次切割
                initial='180 days',  # 初始训练180天
                parallel='processes'
            )

            # 计算性能指标
            df_p = performance_metrics(df_cv)

            # 保存评估结果
            self.evaluation_results['prophet'] = {
                'RMSE': df_p['rmse'].mean(),
                'MAE': df_p['mae'].mean(),
                'MAPE': df_p['mape'].mean(),
                'MSE': df_p['mse'].mean(),
                'Coverage': df_p['coverage'].mean(),
                'horizon': cv_horizon
            }

            print(f"\nProphet模型交叉验证结果:")
            print(f"RMSE: {self.evaluation_results['prophet']['RMSE']:.2f}")
            print(f"MAE: {self.evaluation_results['prophet']['MAE']:.2f}")
            print(f"MAPE: {self.evaluation_results['prophet']['MAPE']:.2%}")

        except Exception as e:
            print(f"Prophet模型评估时出错: {e}")
            # 使用简单的训练测试集评估
            self._simple_prophet_evaluation(model, prophet_data)

    def _simple_prophet_evaluation(self, model, prophet_data):
        """简单的Prophet模型评估"""
        try:
            # 分割训练集和测试集（最后20%作为测试集）
            split_idx = int(len(prophet_data) * 0.8)
            train_data = prophet_data.iloc[:split_idx]
            test_data = prophet_data.iloc[split_idx:]

            # 重新训练模型（仅用于评估）
            params = self.params
            eval_model = Prophet(**params)
            if model.holidays is not None:
                eval_model.holidays = model.holidays

            eval_model.fit(train_data)

            # 创建未来日期数据框
            future = eval_model.make_future_dataframe(periods=len(test_data), freq='D')

            # 预测
            forecast = eval_model.predict(future)

            # 获取预测值（与测试集对齐）
            pred_values = forecast.tail(len(test_data))['yhat'].values
            true_values = test_data['y'].values

            # 计算评估指标
            mse = mean_squared_error(true_values, pred_values)
            mae = mean_absolute_error(true_values, pred_values)
            rmse = np.sqrt(mse)

            # 计算MAPE（避免除零）
            mask = true_values != 0
            if mask.any():
                mape = np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100
            else:
                mape = np.nan

            self.evaluation_results['prophet'] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

            print(f"\nProphet模型简单评估结果:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            if not np.isnan(mape):
                print(f"MAPE: {mape:.2f}%")

        except Exception as e:
            print(f"简单评估Prophet模型时出错: {e}")

    def _prepare_future_regressors(self, model, features_df, future_dates):
        """
        为未来日期准备回归因子数据 - 简化版本
        假设features_df已经按日期聚合

        Args:
            model: 已训练的Prophet模型
            features_df: 历史特征数据（已按日期聚合）
            future_dates: 未来日期序列

        Returns:
            future_df: 包含未来日期和回归因子的DataFrame
        """
        future_df = pd.DataFrame({'ds': future_dates})

        if not hasattr(model, 'extra_regressors') or not model.extra_regressors:
            return future_df

        # 确保日期列是datetime类型
        if '日期' in features_df.columns:
            features_df = features_df.copy()
            features_df['日期'] = pd.to_datetime(features_df['日期'])

        # 处理每个回归因子
        for regressor_name in model.extra_regressors.keys():
            if regressor_name in features_df.columns:
                regressor_data = features_df[regressor_name]

                # 数值型变量
                if pd.api.types.is_numeric_dtype(regressor_data):
                    # 方法1: 使用最近7天的平均值
                    recent_days = min(7, len(regressor_data))
                    recent_avg = regressor_data.tail(recent_days).mean()
                    future_df[regressor_name] = recent_avg

                    # 方法2: 对于有明显周期性的变量，考虑星期几模式
                    if '日期' in features_df.columns and len(features_df) >= 14:
                        # 分析星期几模式
                        weekday_pattern = features_df.groupby(features_df['日期'].dt.weekday)[regressor_name].mean()

                        # 如果有明显的星期几差异（标准差较大），使用星期几模式
                        if weekday_pattern.std() > weekday_pattern.mean() * 0.3:  # 差异较大
                            # 为未来日期填充对应的星期几平均值
                            future_weekdays = future_dates.weekday
                            future_df[regressor_name] = future_weekdays.map(
                                lambda x: weekday_pattern.get(x, recent_avg)
                            )

                # 布尔型变量（如是否促销）
                elif regressor_name.startswith('是否') or regressor_data.dtype == bool:
                    # 计算历史促销频率
                    if len(regressor_data) > 0:
                        promo_rate = regressor_data.mean()
                        # 按概率预测未来促销
                        future_df[regressor_name] = (np.random.rand(len(future_dates)) < promo_rate).astype(int)
                    else:
                        future_df[regressor_name] = 0

                # 其他类型
                else:
                    # 使用最近一天的值
                    if len(regressor_data) > 0:
                        future_df[regressor_name] = regressor_data.iloc[-1]
                    else:
                        future_df[regressor_name] = 0
            else:
                future_df[regressor_name] = 0

        # 填充NaN值
        for col in future_df.columns:
            if col != 'ds' and future_df[col].isna().any():
                if pd.api.types.is_numeric_dtype(future_df[col]):
                    future_df[col] = future_df[col].fillna(0)
                else:
                    future_df[col] = future_df[col].fillna(0)

        return future_df
    def predict_prophet(self, features_df, periods=7, freq='D'):
        """
        使用Prophet模型进行预测

        参数:
        - features_df: 特征数据框
        - periods: 预测未来多少期
        - freq: 预测频率（'D'表示天）

        返回:
        - 预测结果DataFrame
        """
        if 'prophet' not in self.models:
            raise ValueError("Prophet模型尚未训练")

        model = self.models['prophet']

        # 创建未来日期数据框
        last_date = pd.to_datetime(features_df['日期'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=periods, freq=freq)

        # future = pd.DataFrame({'ds': future_dates})
        #
        # # 添加额外回归因子（如果模型中有的话）
        # if hasattr(model, 'extra_regressors') and model.extra_regressors:
        #     # 这里需要根据实际情况为未来日期提供回归因子的值
        #     # 在实际应用中，可能需要基于历史数据或外部预测来提供这些值
        #     for regressor in model.extra_regressors.keys():
        #         # 简单处理：使用最近7天的平均值
        #         if regressor in features_df.columns:
        #             # recent_value = features_df[regressor].tail(7).mean()
        #             recent_value = features_df[regressor].tail(1)
        #             future[regressor] = recent_value
        #         else:
        #             # 如果没有该回归因子的数据，使用0
        #             future[regressor] = 0
        # 准备包含回归因子的未来数据
        future = self._prepare_future_regressors(model, features_df, future_dates)
        # 进行预测
        forecast = model.predict(future)

        # 提取预测结果
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result = result.rename(columns={
            'ds': '预测日期',
            'yhat': '预测销量',
            'yhat_lower': '预测下限',
            'yhat_upper': '预测上限'
        })

        # 格式化日期
        result['预测日期'] = result['预测日期'].dt.strftime('%Y-%m-%d')

        return result

    def plot_prophet_forecast(self, features_df, forecast_periods=1):
        """
        绘制Prophet预测结果

        参数:
        - features_df: 特征数据框
        - forecast_periods: 预测期数

        返回:
        - 交互式Plotly图表
        """
        if 'prophet' not in self.models:
            raise ValueError("Prophet模型尚未训练")

        model = self.models['prophet']

        # 准备历史数据
        prophet_data = self._prepare_prophet_data(features_df, '销售数量')

        if prophet_data is None:
            return None

        # 创建未来数据框并预测
        # future = model.make_future_dataframe(periods=forecast_periods, freq='D')
        last_date = pd.to_datetime(features_df['日期'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=forecast_periods, freq='D')
        # 准备包含回归因子的未来数据
        future = self._prepare_future_regressors(model, features_df, future_dates)
        forecast = model.predict(future)

        # 绘制预测结果
        fig = plot_plotly(model, forecast)

        # 自定义图表样式
        fig.update_layout(
            title='Prophet销售预测',
            xaxis_title='日期',
            yaxis_title='销售数量',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_prophet_components(self, features_df):
        """
        绘制Prophet模型组件（趋势、季节性等）

        参数:
        - features_df: 特征数据框

        返回:
        - 组件分解图
        """
        if 'prophet' not in self.models:
            raise ValueError("Prophet模型尚未训练")

        model = self.models['prophet']

        # 准备历史数据
        prophet_data = self._prepare_prophet_data(features_df, '销售数量')

        if prophet_data is None:
            return None

        # 创建未来数据框并预测
        # future = model.make_future_dataframe(periods=7, freq='D')
        last_date = pd.to_datetime(features_df['日期'].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=7, freq='D')
        # 准备包含回归因子的未来数据
        future = self._prepare_future_regressors(model, features_df, future_dates)
        forecast = model.predict(future)

        # 绘制组件
        fig = plot_components_plotly(model, forecast)

        # 自定义图表样式
        fig.update_layout(
            title='Prophet模型组件分解',
            template='plotly_white'
        )

        return fig

    def get_evaluation_results(self, model_name='prophet'):
        """
        获取评估结果

        参数:
        - model_name: 模型名称

        返回:
        - 评估结果字典
        """
        return self.evaluation_results.get(model_name, {})

    def save_model(self, filepath):
        """
        保存训练好的模型

        参数:
        - filepath: 保存路径
        """
        model_data = {
            'model': self.models.get('prophet'),
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'evaluation_results': self.evaluation_results,
            'feature_columns': self.feature_columns,
            'params': self.params,
            '_data_min_date': getattr(self, '_data_min_date', None),
            '_data_max_date': getattr(self, '_data_max_date', None)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """
        加载训练好的模型

        参数:
        - filepath: 加载路径
        """

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models['prophet'] = model_data['model']
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        self.evaluation_results = model_data['evaluation_results']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data['params']
        self._data_min_date = model_data.get('_data_min_date')
        self._data_max_date = model_data.get('_data_max_date')

        print(f"模型已从 {filepath} 加载")