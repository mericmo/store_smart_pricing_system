from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pio.templates.default = "plotly_white"


class ProphetModel:
    # 在现有代码基础上添加以下方法
    def __init__(self):
        self.models={}
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
            params = {
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

        print("开始训练Prophet模型...")

        # 由于Prophet需要特定的数据格式，我们需要重新准备数据
        # Prophet要求两列：ds (datetime类型) 和 y (数值类型)

        # 准备Prophet数据
        prophet_data = self._prepare_prophet_data(features_df, target_col)

        if prophet_data is None or len(prophet_data) < 14:  # 至少需要2周数据
            print(f"数据不足或格式不正确，无法训练Prophet模型")
            return None

        # 创建Prophet模型
        model = Prophet(**params)

        # 添加中国的节假日
        self._add_chinese_holidays(model)

        # 添加额外回归因子
        self._add_regressors(model, features_df)

        # 训练模型
        model.fit(prophet_data)

        # 保存模型
        self.models['prophet'] = model

        # 进行交叉验证评估
        self._evaluate_prophet_model(model, prophet_data)

        print("Prophet模型训练完成!")

        return model

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
            df['日期'] = pd.to_datetime(df['日期'])

            # 按日期聚合数据（因为Prophet处理的是时间序列，通常按天）
            # 如果需要按门店和商品分别预测，这里需要调整
            if '门店编码' in df.columns and '商品编码' in df.columns:
                # 按日期、门店、商品分组
                agg_data = df.groupby(['日期', '门店编码', '商品编码'])[target_col].sum().reset_index()
                # 这里简化处理，取所有门店和商品的总和，实际应用中可能需要分别建模
                # 或者使用分层预测
                prophet_data = df.groupby('日期')[target_col].sum().reset_index()
            else:
                prophet_data = df.groupby('日期')[target_col].sum().reset_index()
            # df['日期时间'] = pd.to_datetime(df['日期']) + pd.to_timedelta(df['小时'], unit='h')
            # prophet_data = df.groupby('日期时间')[target_col].sum().reset_index()
            # 重命名列以符合Prophet格式
            prophet_data = prophet_data.rename(columns={
                '日期': 'ds',
                target_col: 'y'
            })

            # 确保y没有负值（如果有的话，Prophet可以处理但建议调整）
            prophet_data['y'] = prophet_data['y'].clip(lower=0)

            # 按日期排序
            prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)

            print(f"Prophet数据准备完成，共{len(prophet_data)}天数据")
            print(f"日期范围: {prophet_data['ds'].min()} 到 {prophet_data['ds'].max()}")

            return prophet_data

        except Exception as e:
            print(f"准备Prophet数据时出错: {e}")
            return None

    def _add_chinese_holidays(self, model):
        """添加中国节假日"""
        chinese_holidays = pd.DataFrame({
            'holiday': 'chinese_holiday',
            'ds': pd.to_datetime([
                # 春节（农历正月初一，这里简化使用公历近似日期）
                '2023-01-22', '2024-02-10', '2025-01-29',
                # 清明节
                '2023-04-05', '2024-04-04', '2025-04-04',
                # 劳动节
                '2023-05-01', '2024-05-01', '2025-05-01',
                # 端午节
                '2023-06-22', '2024-06-10', '2025-05-31',
                # 中秋节
                '2023-09-29', '2024-09-17', '2025-10-06',
                # 国庆节
                '2023-10-01', '2024-10-01', '2025-10-01',
            ]),
            'lower_window': -2,  # 节前2天
            'upper_window': 2,  # 节后2天
        })

        # 添加节假日到模型
        model.holidays = chinese_holidays

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
        weather_cols = ['温度', '最高温度', '最低温度', '温度差', '是否下雨', '天气严重程度']
        for col in weather_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        # 时间相关特征
        time_cols = ['是否周末', '是否节假日', '月份', '星期几', '季度']
        for col in time_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        # 促销相关特征
        promo_cols = ['是否有促销', '折扣率', '促销天数']
        for col in promo_cols:
            if col in features_df.columns:
                regressor_cols.append(col)

        # 添加回归因子
        for col in regressor_cols:
            try:
                # 按日期聚合回归因子（取平均值）
                if col in features_df.columns:
                    regressor_data = features_df.groupby('日期')[col].mean().reset_index()
                    regressor_data = regressor_data.rename(columns={'日期': 'ds', col: col})

                    # 添加回归因子到模型
                    model.add_regressor(col)
                    print(f"已添加回归因子: {col}")
            except Exception as e:
                print(f"添加回归因子 {col} 时出错: {e}")

    def _evaluate_prophet_model(self, model, prophet_data, cv_horizon='30 days'):
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
            # 分割训练集和测试集（最后30天作为测试集）
            split_idx = int(len(prophet_data) * 0.8)
            train_data = prophet_data.iloc[:split_idx]
            test_data = prophet_data.iloc[split_idx:]

            # 重新训练模型（仅用于评估）
            eval_model = Prophet(**model.get_params())
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

        future = pd.DataFrame({'ds': future_dates})

        # 添加额外回归因子（如果模型中有的话）
        if hasattr(model, 'extra_regressors') and model.extra_regressors:
            # 这里需要根据实际情况为未来日期提供回归因子的值
            # 在实际应用中，可能需要基于历史数据或外部预测来提供这些值
            for regressor in model.extra_regressors.keys():
                # 简单处理：使用最近7天的平均值
                if regressor in features_df.columns:
                    recent_value = features_df[regressor].tail(7).mean()
                    future[regressor] = recent_value
                else:
                    # 如果没有该回归因子的数据，使用0
                    future[regressor] = 0

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

    def plot_prophet_forecast(self, features_df, forecast_periods=30):
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
        future = model.make_future_dataframe(periods=forecast_periods, freq='D')
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
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)

        # 绘制组件
        fig = plot_components_plotly(model, forecast)

        # 自定义图表样式
        fig.update_layout(
            title='Prophet模型组件分解',
            template='plotly_white'
        )

        return fig