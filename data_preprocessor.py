# data_preprocessor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import LocalOutlierFactor
# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from utils.time_utils import time_to_custom_min_slot
from utils.common import save_to_csv


class DataPreprocessor:
    """
    负责原始数据的清洗、格式标准化、基础字段处理
    不涉及复杂的特征计算
    """

    def __init__(self, product_code=None, store_code=None):
        self.processed_data = {}
        self.store_code = store_code
        self.product_code = product_code

    def preprocess_sales_data(self, hsr_df):
        """销售数据基础预处理"""
        # 1. 日期格式标准化
        # hsr_df['日期'] = pd.to_datetime(hsr_df['日期'])

        # 1. 按门店、商品进行过滤
        hsr_df = self._filter_store_product(hsr_df)

        # 2. 基础字段处理,设置统计时间窗口
        hsr_df = self._process_basic_sales_fields(hsr_df, gap=60)

        # 3. 创建按小时或30min聚合数据
        daily_sales = self._create_hour_aggregation(hsr_df)
        # 3. 创建每日聚合（商品×日期级别）
        # daily_sales = self._create_daily_aggregation(hsr_df)

        # 4. 填补缺失日期
        # daily_sales = self._fill_missing_dates(daily_sales)

        # daily_sales = self._fill_missing_dates_hour(daily_sales)
        # daily_sales = self._fill_missing_dates_by_timewindow_optimized(daily_sales)
        daily_sales = self.remove_outliers_lof(daily_sales, columns=['销售数量'], contamination=0.01)
        save_to_csv(daily_sales)
        return daily_sales

    def _filter_store_product(self, df):

        store_mask = df['门店编码'] == self.store_code
        product_mask = df['商品编码'] == self.product_code
        qty_mask = df['销售数量'] > 0
        channel_mask = df['渠道名称'] == '线下销售'
        df = df[store_mask & product_mask & qty_mask & channel_mask].drop(
            ['会员id', '流水单号', '平台触点名称', '渠道名称', '小类编码'], axis=1)
        if len(df) < 50:
            raise Exception(f"商品{self.product_code}数据两太少,记录数：{len(df)}。")
        save_to_csv(df)
        return df

    def _process_basic_sales_fields(self, df, gap: int = 60):
        """处理基础销售字段"""
        # 价格计算 - 使用售价字段，不需要重新计算
        # 确保售价是数值类型
        df['日期'] = pd.to_datetime(df['日期'])
        # 计算日期序号（从数据开始日期算起）
        min_date = df['日期'].min()
        df['日期序号'] = (df['日期'] - min_date).dt.days + 1

        df['售价'] = pd.to_numeric(df['售价'], errors='coerce')
        if "销售净额" in df.columns and "销售金额" in df.columns:
            # 确保销售数量为数值（若有非数字或空值会变为 NaN）
            df["销售数量"] = pd.to_numeric(df["销售数量"], errors="coerce")
            # 确保金额列为数值
            df['销售净额'] = pd.to_numeric(df['销售净额'], errors="coerce")

            # 确保金额列为数值
            df["销售金额"] = pd.to_numeric(df["销售金额"], errors="coerce")
            # 折扣率
            df['实际折扣率'] = np.where(
                df["销售金额"] > 0,
                1 - df['折扣金额'] / df["销售金额"],
                np.nan
            )
        # df['时间窗口'] = df['交易时间'].dt.floor('30min')
        if gap == 60:
            df['时间窗口'] = df['交易时间'].dt.hour
        else:
            df['时间窗口'] = time_to_custom_min_slot(df['交易时间'], gap)

        # 基础分类字段
        # df['商品小类'] = df['小类编码'].astype('category')
        df['是否促销'] = (df['折扣类型'] != 'n-无折扣促销').astype(int)
        df = df.dropna()
        save_to_csv(df)
        return df

    def _create_daily_aggregation(self, df):
        """创建每日聚合数据"""
        # 先处理销售数量为负的情况（退货）
        # df = self._handle_negative_sales(df)

        # daily_agg = df.groupby(['商品编码', '日期']).agg({
        #     '销售数量': 'sum',
        #     '销售金额': 'sum',
        #     '售价': 'mean',
        #     '是否促销': 'sum',  # 促销天数
        #     '渠道名称': 'nunique',  # 销售渠道数
        #     '会员id': lambda x: x.notna().sum()  # 会员购买次数
        # }).reset_index()
        #
        # daily_agg = daily_agg.rename(columns={
        #     '是否促销': '促销天数',
        #     '渠道名称': '销售渠道数',
        #     '会员id': '会员购买次数'
        # })
        daily_agg = df.groupby(['门店编码', '商品编码', '商品名称', '日期']).agg({
            '销售数量': 'sum',
            '销售金额': 'sum',
            '售价': 'mean',
            '是否促销': 'mean',
            '时间窗口': 'mean',

        }).reset_index()

        return self._create_count_aggregation(daily_agg)

    def _create_count_aggregation(self, daily_agg: pd.DataFrame):
        # 平均售价
        daily_agg["平均售价"] = np.where(
            daily_agg["销售数量"] > 0,
            (daily_agg['销售金额']-daily_agg['折扣金额']) / daily_agg["销售数量"],
            np.nan
        )
        daily_agg["实际折扣率"] = np.where(
            daily_agg["销售金额"] > 0,
            1 - daily_agg['折扣金额'] / daily_agg["销售金额"],
            np.nan
        )
        # 处理除以0的情况
        daily_agg['实际折扣率'] = daily_agg['实际折扣率'].fillna(1.0)
        daily_agg['平均售价'] = daily_agg['平均售价'].fillna(daily_agg["售价"])
        # 限制折扣率在合理范围 [0, 1]
        # daily_agg['实际折扣率'] = daily_agg['实际折扣率'].clip(0, 1)
        save_to_csv(daily_agg)
        return daily_agg

    def _create_hour_aggregation(self, df):
        """创建每日聚合数据"""
        # 先处理销售数量为负的情况（退货）
        # df = self._handle_negative_sales(df)

        agg_dict = {
            '销售数量': 'sum',
            '销售金额': 'sum',
            '售价': 'mean',
            '是否促销': 'mean',
            '折扣金额': 'sum',
        }
        # if '实际折扣率' in df.columns:
        #     agg_dict['实际折扣率'] = 'mean'
        # else:
        #     # 如果没有实际折扣率列，创建一个默认列
        #     df['实际折扣率'] = 1.0
        #     agg_dict['实际折扣率'] = 'mean'
        #
        # if '是否促销' in df.columns:
        #     agg_dict['是否促销'] = 'mean'
        # else:
        #     # 创建默认的是否折扣列
        #     df['是否促销'] = 0.0
        #     agg_dict['是否促销'] = 'mean'
        group_by_key = ['门店编码', '商品编码', '商品名称', '日期', '时间窗口']
        # 使用修复的agg字典
        daily_agg = df.groupby(group_by_key).agg(agg_dict).reset_index()
        save_to_csv(daily_agg)
        return self._create_count_aggregation(daily_agg)
        # try:
        #     # 使用修复的agg字典
        #     daily_agg = df.groupby(group_by_key).agg(agg_dict).reset_index()
        #     daily_agg['实际折扣率'] = 1 - daily_agg['折扣金额'] / daily_agg['销售金额']
        #     # 处理除以0的情况（当销售金额和折扣金额都为0时）
        #     daily_agg['实际折扣率'] = daily_agg['实际折扣率'].fillna(1)
        #     # 限制折扣率在合理范围 [0, 1]
        #     daily_agg['实际折扣率'] = daily_agg['实际折扣率'].clip(0, 1)
        # except Exception as e:
        #     # 如果仍然出错，使用更简单的聚合方式
        #     print(f"警告: 标准groupby失败, 使用替代方法: {e}")
        #     daily_agg = pd.DataFrame({
        #         '时间窗口': df['时间窗口'].unique(),
        #         '销售数量': df.groupby(group_by_key)['销售数量'].sum().values,
        #         '平均售价': df.groupby(group_by_key)['平均售价'].mean().values,
        #         '销售金额': df.groupby(group_by_key)['销售金额'].mean().values,
        #         '售价': df.groupby(group_by_key)['售价'].mean().values,
        #     })
        #
        #     if '实际折扣率' in df.columns:
        #         daily_agg['实际折扣率'] = df.groupby(group_by_key)['实际折扣率'].mean().values
        #     else:
        #         daily_agg['实际折扣率'] = 1.0
        #
        #     if '是否促销' in df.columns:
        #         daily_agg['是否促销'] = df.groupby(group_by_key)['是否促销'].mean().values
        #     else:
        #         daily_agg['是否促销'] = 0.0
        # daily_agg.to_csv('_create_hour_aggregation.csv', encoding='utf-8')
        # return daily_agg

    # def _handle_negative_sales(self, df):
    #     """处理负销售数量（退货情况）"""
    #     # 记录退货数量用于分析
    #     returns = df[df['销售数量'] < 0].copy()
    #     if len(returns) > 0:
    #         print(f"发现 {len(returns)} 条退货记录")
    #
    #     # 对于训练数据，我们可以选择：
    #     # 1. 移除退货记录
    #     # 2. 将退货视为0销售
    #     # 这里选择移除退货记录，因为负值会影响模型训练
    #     df = df[df['销售数量'] >= 0]
    #
    #     return df

    # def _fill_missing_dates(self, daily_sales):
    #     """填补缺失日期，创建完整的时间序列"""
    #     # 确保日期格式正确
    #     daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
    #
    #     date_range = pd.date_range(
    #         start=daily_sales['日期'].min(),
    #         end=daily_sales['日期'].max(),
    #         freq='D'
    #     )
    #
    #     # 为每个商品创建完整的时间序列
    #     all_products = daily_sales['商品编码'].unique()
    #     full_index = pd.MultiIndex.from_product(
    #         [all_products, date_range],
    #         names=['商品编码', '日期']
    #     )
    #
    #     daily_sales_full = daily_sales.set_index(['商品编码', '日期']).reindex(full_index).reset_index()
    #
    #     # 填充缺失值
    #     daily_sales_full['销售数量'] = daily_sales_full['销售数量'].fillna(0)
    #     daily_sales_full['销售金额'] = daily_sales_full['销售金额'].fillna(0)
    #
    #     daily_sales_full['售价'] = daily_sales_full.groupby('商品编码')['售价'].transform(
    #         lambda x: x.fillna(x.mean())
    #     )
    #     daily_sales_full['促销天数'] = daily_sales_full['促销天数'].fillna(0)
    #     daily_sales_full['销售渠道数'] = daily_sales_full['销售渠道数'].fillna(0)
    #     daily_sales_full['会员购买次数'] = daily_sales_full['会员购买次数'].fillna(0)
    #
    #     # 填充商品名称 - 使用groupby来填充，避免merge导致的重复列名问题
    #     daily_sales_full['商品名称'] = daily_sales_full.groupby('商品编码')['商品名称'].transform(
    #         lambda x: x.fillna(method='ffill').fillna(method='bfill')
    #     )
    #
    #     return daily_sales_full

    def _fill_missing_dates_hour(self, daily_sales):
        """填补缺失日期，创建完整的时间序列"""
        # 确保日期格式正确
        daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])

        date_range = daily_sales['日期'].unique()
        # 为每个商品创建完整的时间序列
        group_by_key = ['商品编码', '日期', '时间窗口']
        all_products = daily_sales['商品编码'].unique()
        # time_window = daily_sales['时间窗口'].unique()
        # 获取时间窗口范围（从最小值到最大值）
        time_window_min = int(daily_sales['时间窗口'].min())
        time_window_max = int(daily_sales['时间窗口'].max())
        time_window_range = range(time_window_min, time_window_max + 1)  # 包含最大值

        full_index = pd.MultiIndex.from_product(
            [all_products, date_range, time_window_range],
            names=group_by_key
        )

        daily_sales_full = daily_sales.set_index(group_by_key).reindex(full_index).reset_index()

        daily_sales_full['时间窗口'] = daily_sales_full['时间窗口'].astype(int)
        # 填充缺失值
        daily_sales_full['销售数量'] = daily_sales_full['销售数量'].fillna(0)
        daily_sales_full['销售金额'] = daily_sales_full['销售金额'].fillna(0)
        # 填充商品名称 - 使用groupby来填充，避免merge导致的重复列名问题
        # daily_sales_full['商品名称'] = daily_sales_full.groupby('商品编码')['商品名称'].transform(
        #     lambda x: x.fillna(method='ffill').fillna(method='bfill')
        # )

        fill_cols = ['售价', '平均售价', '是否促销', '实际折扣率', '商品名称', '门店编码']
        for col in fill_cols:
            daily_sales_full[col] = daily_sales_full.groupby(['商品编码', '日期'])[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
        # 还有空值的话，用默认值填充
        daily_sales_full['售价'] = daily_sales_full.groupby('商品编码')['售价'].transform(
            lambda x: x.fillna(x.mean())
        )
        daily_sales_full['平均售价'] = daily_sales_full.groupby('商品编码')['平均售价'].transform(
            lambda x: x.fillna(x.mean())
        )
        daily_sales_full['是否促销'] = daily_sales_full['是否促销'].fillna(0)

        # 排序
        daily_sales_full = daily_sales_full.sort_values(['商品编码', '日期', '时间窗口']).reset_index(drop=True)

        return daily_sales_full

    def preprocess_weather_data(self, weather_df):
        """天气数据基础预处理"""
        # 重命名列以匹配中文
        weather_df = weather_df.rename(columns={
            'date': '日期',
            'high': '最高温度',
            'low': '最低温度'
        })

        # 日期格式标准化
        weather_df['日期'] = pd.to_datetime(weather_df['日期'])

        # 数值字段处理
        weather_df['最高温度'] = pd.to_numeric(weather_df['最高温度'], errors='coerce')
        weather_df['最低温度'] = pd.to_numeric(weather_df['最低温度'], errors='coerce')

        # 计算平均温度
        weather_df['温度'] = (weather_df['最高温度'] + weather_df['最低温度']) / 2

        # 计算温差
        # weather_df['温度差'] = weather_df['最高温度'] - weather_df['最低温度']
        # weather_df['天气描述'] = weather_df['code_day'].astype(str)

        # 天气严重程度评分
        def get_weather_severity(code):
            if pd.isna(code):
                return 0

            code_str = str(code).zfill(2)
            severity_map = {
                '01': 0,  # 晴
                '02': 1,  # 多云
                '03': 2,  # 阴
                '04': 3,  # 阵雨
                '07': 4,  # 小雨
                '08': 5,  # 中雨
                '09': 6,  # 大雨
                '10': 7,  # 暴雨
                '13': 4,  # 阵雪
                '14': 5,  # 小雪
                '15': 6,  # 中雪
                '16': 7,  # 大雪
                '18': 3,  # 雾
                '53': 4,  # 霾
            }
            return severity_map.get(code_str, 0)

        if 'code_day' in weather_df.columns:
            weather_df['天气严重程度'] = weather_df['code_day'].apply(get_weather_severity)

        # 填充缺失值
        weather_df['温度'] = weather_df['温度'].fillna(method='ffill').fillna(method='bfill')
        weather_df['最高温度'] = weather_df['最高温度'].fillna(method='ffill').fillna(method='bfill')
        weather_df['最低温度'] = weather_df['最低温度'].fillna(method='ffill').fillna(method='bfill')

        # 由于天气数据中没有降雨量，我们创建一个模拟的降雨量字段
        # 在实际项目中，你需要真实的降雨量数据
        weather_df['降雨量'] = 0  # 默认为0
        weather_df = weather_df[['日期', '温度', '最高温度', '最低温度', '降雨量', '天气严重程度']].sort_values(
            ['日期']).reset_index(drop=True)
        save_to_csv(weather_df)
        return weather_df

    def preprocess_calendar_data(self, calendar_df):
        """日历数据基础预处理"""
        # 销售表里面把是否周末已经识别出来了，这里不用加，否则会导致冲突
        # 重命名列以匹配中文
        calendar_df = calendar_df.rename(columns={
            'date': '日期',
            'holiday_legal': '是否节假日'
        })

        # 日期格式标准化
        calendar_df['日期'] = pd.to_datetime(calendar_df['日期'].astype(str))

        # 处理节假日字段
        calendar_df['是否节假日'] = calendar_df['是否节假日'].fillna(0).astype(int)

        print(calendar_df.head(1))
        # 创建节假日类型字段（简化处理）
        calendar_df['节假日类型'] = calendar_df['是否节假日'].apply(
            lambda x: '法定节假日' if x == 1 else '普通日'
        )
        calendar_df = calendar_df[['日期', '是否节假日', '节假日类型']].sort_values(['日期']).reset_index(drop=True)
        save_to_csv(calendar_df)
        return calendar_df

    def get_data_summary(self):
        """获取数据摘要信息"""
        summary = {}
        for data_type, data in self.processed_data.items():
            summary[data_type] = {
                '数据量': len(data),
                '列数': len(data.columns),
                '时间范围': f"{data['日期'].min()} 到 {data['日期'].max()}" if '日期' in data.columns else '无日期列',
                '缺失值统计': data.isnull().sum().to_dict()
            }
        return summary

    def remove_outliers_lof(self, df, columns=None, contamination=0.01, n_neighbors=50,
                            metric='minkowski', p=2, **kwargs):
        """
        使用LOF检测局部异常值
        支持多列（多维）异常值检测

        参数:
        ----------
        df : DataFrame
            输入数据
        columns : list or None
            要处理的列列表，如果为None则处理所有数值列
        contamination : float, default=0.1
            异常值比例估计 (0 < contamination <= 0.5)
        n_neighbors : int, default=20
            邻居数量，通常设置与数据维度相关
        metric : str, default='minkowski'
            距离度量方法
        p : int, default=2
            Minkowski距离的幂参数（p=2为欧氏距离）
        **kwargs :
            其他传递给LocalOutlierFactor的参数

        返回:
        -------
        df_clean : DataFrame
            去除异常值后的数据
        df_outliers : DataFrame
            被标记为异常值的数据
        """
        df_clean = df.copy()

        if columns is None:
            # 默认处理所有数值列
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            print("警告：没有找到数值列！")
            return df_clean, pd.DataFrame()

        print(f"使用LOF检测 {len(columns)} 个特征的异常值:")
        print(f"特征: {columns}")

        # 处理缺失值 - 使用中位数填充
        df_selected = df[columns].copy()
        missing_counts = df_selected.isnull().sum().sum()

        if missing_counts > 0:
            print(f"填充 {missing_counts} 个缺失值...")
            df_filled = df_selected.fillna(df_selected.median())
        else:
            df_filled = df_selected

        # 标准化数据（对LOF很重要，因为基于距离）

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filled)

        # 自动调整n_neighbors（如果未指定）
        if n_neighbors == 'auto':
            n_neighbors = min(20, max(10, len(df_scaled) // 10))
            print(f"自动设置n_neighbors为: {n_neighbors}")

        # 训练LOF
        try:
            lof = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=n_neighbors,
                metric=metric,
                p=p,
                **kwargs
            )

            predictions = lof.fit_predict(df_scaled)

        except Exception as e:
            print(f"LOF训练错误: {e}")
            # 回退到更简单的参数
            print("使用简化参数重试...")
            lof = LocalOutlierFactor(
                contamination=min(0.2, contamination),
                n_neighbors=min(10, len(df_scaled) - 1),
                metric='euclidean'
            )
            predictions = lof.fit_predict(df_scaled)

        # -1 表示异常值，1 表示正常值
        mask = predictions == 1

        print(f"\nLOF检测结果:")
        print(f"原始数据: {len(df)} 条")
        print(f"正常值: {mask.sum()} 条")
        print(f"异常值: {(~mask).sum()} 条")
        print(f"异常值比例: {(~mask).sum() / len(df) * 100:.2f}%")
        abnormal_data = df_clean[~mask]
        save_to_csv(abnormal_data)
        # 返回去除异常值和异常值本身
        return df_clean[mask].sort_values(['日期', '时间窗口']).reset_index(drop=True)
