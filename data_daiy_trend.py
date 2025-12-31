import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# matplotlib.use("Agg")
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


product_code = '3160860'  # 请替换为实际的产品编码

# 尝试读取数据
try:
    features_df = pd.read_csv('data/historical_transactions.csv', encoding='utf-8', parse_dates=['日期'],
                              dtype={'商品编码': str})
    features_df = features_df[(features_df['商品编码'] == product_code) &
                              (features_df['销售数量'] > 0) &
                              (features_df['渠道名称'] == '线下销售')]
    print("共计{}条数据".format(len(features_df)))
    # 检查是否有数据
    if len(features_df) == 0:
        print(f"警告: 没有找到产品编码为 '{product_code}' 的线下销售数据")
        # 创建一个空的图表显示提示
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"没有找到产品编码 '{product_code}' 的线下销售数据",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f"产品 {product_code} 销售分析")
        plt.show()
    else:
        # 数据处理
        df = features_df.groupby(['日期']).agg({
            '销售数量': 'sum',
        }).reset_index()

        # 设置日期为索引
        df.set_index('日期', inplace=True)

        # 确保时间序列连续
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(date_range)
        df['销售数量'] = df['销售数量'].fillna(0)

        # 创建图表 - 使用更简单的布局避免版本兼容性问题
        fig = plt.figure(figsize=(15, 12))

        # 1. 原始序列
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df['销售数量'], label='原始序列', color='blue', linewidth=1.5)
        ax1.set_title(f'时间序列趋势 - 产品编码: {product_code}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('销售数量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 移动平均趋势
        ax2 = plt.subplot(3, 1, 2)
        window_size = min(7, len(df) // 4)
        if window_size > 1:
            df['移动平均'] = df['销售数量'].rolling(window=window_size, center=True, min_periods=1).mean()
            ax2.plot(df.index, df['销售数量'], label='原始序列', color='blue', alpha=0.5, linewidth=1)
            ax2.plot(df.index, df['移动平均'], label=f'{window_size}天移动平均', color='red', linewidth=2)
            ax2.set_title('移动平均趋势分析', fontsize=12)
        else:
            ax2.plot(df.index, df['销售数量'], label='原始序列', color='blue', linewidth=1.5)
            ax2.set_title('销售序列（数据不足计算移动平均）', fontsize=12)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('销售数量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 周内销售模式
        ax3 = plt.subplot(3, 1, 3)

        # 添加星期信息
        df_copy = df.copy()
        df_copy['星期'] = df_copy.index.dayofweek

        # 计算周内平均销售
        weekly_avg = df_copy.groupby('星期')['销售数量'].mean()

        days_chinese = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

        if len(weekly_avg) > 0:
            # 确保有7天的数据
            complete_data = []
            for day in range(7):
                if day in weekly_avg.index:
                    complete_data.append(weekly_avg[day])
                else:
                    complete_data.append(0)

            bars = ax3.bar(range(7), complete_data, alpha=0.7, color='teal')
            ax3.set_title('周内销售模式', fontsize=12)
            ax3.set_xlabel('星期')
            ax3.set_ylabel('平均销售数量')
            ax3.set_xticks(range(7))
            ax3.set_xticklabels(days_chinese, rotation=45)

            # 在柱状图上添加数值标签
            for i, v in enumerate(complete_data):
                if v > 0:
                    ax3.text(i, v + max(complete_data) * 0.02, f'{v:.1f}',
                             ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, '没有足够的周内数据',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('周内销售模式', fontsize=12)

        ax3.grid(True, alpha=0.3, axis='y')

        # 调整布局
        plt.tight_layout()
        # plt.show()
        fig.savefig(f'item_{product_code}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # 重要：关闭图表释放内存
        # 输出统计信息
        print("=" * 60)
        print("产品销售统计分析")
        print("=" * 60)
        print(f"产品编码: {product_code}")
        print(f"数据时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
        print(f"总天数: {len(df)}")
        print(f"有销售天数: {(df['销售数量'] > 0).sum()}")
        print(f"零销售天数: {(df['销售数量'] == 0).sum()}")
        print(f"总销售量: {df['销售数量'].sum():.0f}")

        # 正销售统计
        positive_sales = df['销售数量'][df['销售数量'] > 0]
        if len(positive_sales) > 0:
            print(f"平均日销量(仅正销量): {positive_sales.mean():.2f}")
            print(f"日销量标准差(仅正销量): {positive_sales.std():.2f}")
            print(f"日销量最大值: {positive_sales.max():.0f}")
            print(f"日销量最小值(正销量): {positive_sales.min():.0f}")
            print(f"日销量中位数(仅正销量): {positive_sales.median():.2f}")

            # 计算周内统计
            df_copy = df.copy()
            df_copy['星期'] = df_copy.index.dayofweek
            df_copy['星期名称'] = df_copy['星期'].map(lambda x: days_chinese[x])

            print("\n周内销售统计:")
            weekday_stats = df_copy.groupby('星期名称')['销售数量'].agg(['count', 'sum', 'mean', 'std'])
            print(weekday_stats.round(2))

            # 月度统计
            df_copy['月份'] = df_copy.index.month
            month_names = {1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
                           7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}
            df_copy['月份名称'] = df_copy['月份'].map(month_names)

            print("\n月度销售统计:")
            monthly_stats = df_copy.groupby('月份名称')['销售数量'].agg(['count', 'sum', 'mean', 'std'])
            print(monthly_stats.round(2))
        else:
            print("没有正销量记录")

        print("\n销售日分布:")
        print(df['销售数量'].value_counts().sort_values().sort_index().head(20))

        print("=" * 60)

except FileNotFoundError:
    print("错误: 找不到数据文件 'data/historical_transactions.csv'")
    print("请确保文件路径正确，且文件存在")
except KeyError as e:
    print(f"错误: 数据文件中缺少必要的列: {e}")
    print("请确保CSV文件包含以下列: '日期', '商品编码', '销售数量', '渠道名称'")
except Exception as e:
    print(f"发生未知错误: {e}")
    import traceback

    traceback.print_exc()


# 添加一个简单的销售趋势分析函数
def analyze_sales_trend(df, product_code):
    """分析销售趋势"""
    if len(df) == 0:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 销售时间序列
    axes[0, 0].plot(df.index, df['销售数量'], color='blue', linewidth=1)
    axes[0, 0].set_title(f'{product_code} 销售时间序列')
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('销售数量')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 累计销售
    df['累计销售'] = df['销售数量'].cumsum()
    axes[0, 1].plot(df.index, df['累计销售'], color='green', linewidth=2)
    axes[0, 1].set_title(f'{product_code} 累计销售')
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('累计销售数量')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 销售分布直方图
    if (df['销售数量'] > 0).sum() > 0:
        positive_sales = df['销售数量'][df['销售数量'] > 0]
        axes[1, 0].hist(positive_sales, bins=min(20, len(positive_sales)),
                        alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title(f'{product_code} 销售数量分布')
        axes[1, 0].set_xlabel('销售数量')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. 月度销售柱状图
    monthly_sales = df.resample('M')['销售数量'].sum()
    if len(monthly_sales) > 0:
        months = [d.strftime('%Y-%m') for d in monthly_sales.index]
        axes[1, 1].bar(months, monthly_sales.values, alpha=0.7, color='purple')
        axes[1, 1].set_title(f'{product_code} 月度销售')
        axes[1, 1].set_xlabel('月份')
        axes[1, 1].set_ylabel('销售数量')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return fig


# 如果前面成功处理了数据，可以进行进一步分析
try:
    if 'df' in locals() and len(df) > 0:
        print("\n正在进行深入销售趋势分析...")
        analyze_sales_trend(df, product_code)
except Exception as e:
    print(f"趋势分析失败: {e}")