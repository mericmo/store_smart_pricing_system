import pandas as pd


def analyze_transaction_data(df):
    """
    分析交易数据，计算关键指标

    参数:
    df: pandas DataFrame，包含交易数据

    返回:
    dict: 包含分析结果的字典
    """
    # 确保数值列是数值类型
    numeric_cols = ['销售金额', '折扣金额', '销售数量']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确保日期列是日期类型
    if '交易时间' in df.columns:
        df['交易时间'] = pd.to_datetime(df['交易时间'])
        df['日期'] = df['交易时间'].dt.date
    elif '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期']).dt.date

    # 1. 总销售额
    total_sales = df['销售金额'].sum()

    # 2. 总折扣金额
    total_discount = df['折扣金额'].sum()

    # 3. 实际销售额（销售净额）
    if '销售净额' in df.columns:
        total_net_sales = df['销售净额'].sum()
    else:
        total_net_sales = total_sales - total_discount

    # 4. 订单数量统计（按流水单号去重）
    if '流水单号' in df.columns:
        # 每日订单数量
        daily_orders = df.groupby('日期')['流水单号'].nunique()
        daily_orders_count = daily_orders.to_dict()

        # 总订单数量
        total_orders = df['流水单号'].nunique()

        # 有订单的天数
        days_with_orders = len(daily_orders)

        # 日均订单数量
        if days_with_orders > 0:
            avg_daily_orders = total_orders / days_with_orders
        else:
            avg_daily_orders = 0
    else:
        daily_orders_count = {}
        total_orders = 0
        days_with_orders = 0
        avg_daily_orders = 0

    # 5. 平均折扣率
    if total_sales > 0:
        avg_discount_rate = total_discount / total_sales
    else:
        avg_discount_rate = 0

    # 6. 折扣类型分布
    if '折扣类型' in df.columns:
        discount_type_dist = df['折扣类型'].value_counts().to_dict()
    else:
        discount_type_dist = {}

    # 7. 渠道销售额分布
    if '渠道名称' in df.columns:
        channel_sales = df.groupby('渠道名称')['销售金额'].sum().sort_values(ascending=False)
        channel_sales_dist = channel_sales.to_dict()
    else:
        channel_sales_dist = {}

    # 8. 商品销售Top10
    if '商品名称' in df.columns:
        top_10_products = df.groupby('商品名称')['销售金额'].sum().sort_values(ascending=False).head(10)
        top_10_products_dict = top_10_products.to_dict()
    else:
        top_10_products_dict = {}

    # 编译分析结果
    analysis_results = {
        '总销售额': round(total_sales, 2),
        '总折扣金额': round(total_discount, 2),
        '实际销售额': round(total_net_sales, 2),
        '折扣率': round(avg_discount_rate * 100, 2),  # 百分比
        '总订单数量': total_orders,
        '日均订单数量': round(avg_daily_orders, 2),
        '有订单的天数': days_with_orders,
        '每日订单数量': daily_orders_count,
        '折扣类型分布': discount_type_dist,
        '渠道销售额分布': channel_sales_dist,
        '热销商品Top10': top_10_products_dict
    }

    return analysis_results


def print_analysis_summary(results):
    """打印分析结果摘要"""
    print("=" * 60)
    print("交易数据分析报告")
    print("=" * 60)
    print(f"总销售额: ¥{results['总销售额']:.2f}")
    print(f"总折扣金额: ¥{results['总折扣金额']:.2f}")
    print(f"实际销售额: ¥{results['实际销售额']:.2f}")
    print(f"平均折扣率: {results['折扣率']:.2f}%")
    print(f"总订单数量: {results['总订单数量']}")
    print(f"有订单的天数: {results['有订单的天数']}")
    print(f"日均订单数量: {results['日均订单数量']:.2f}")

    print("\n" + "-" * 60)
    print("折扣类型分布:")
    for discount_type, count in results['折扣类型分布'].items():
        print(f"  {discount_type}: {count}次")

    print("\n" + "-" * 60)
    print("渠道销售额分布:")
    for channel, sales in results['渠道销售额分布'].items():
        print(f"  {channel}: ¥{sales:.2f}")

    print("\n" + "-" * 60)
    print("热销商品Top10:")
    for i, (product, sales) in enumerate(results['热销商品Top10'].items(), 1):
        print(f"  {i}. {product}: ¥{sales:.2f}")


# 使用示例
if __name__ == "__main__":
    # 读取CSV文件
    # 注意：根据您的文件编码和分隔符调整参数
    df = pd.read_csv('./K5.交易流水明细表2026-01-13_9_49_12_xian.csv', encoding='utf-8')

    # 分析数据
    results = analyze_transaction_data(df)

    # 打印结果
    print_analysis_summary(results)

    # 保存详细结果到CSV
    detailed_results = pd.DataFrame([
        ['总销售额', results['总销售额']],
        ['总折扣金额', results['总折扣金额']],
        ['实际销售额', results['实际销售额']],
        ['平均折扣率(%)', results['折扣率']],
        ['总订单数量', results['总订单数量']],
        ['日均订单数量', results['日均订单数量']],
        ['有订单的天数', results['有订单的天数']]
    ], columns=['指标', '值'])

    detailed_results.to_csv('交易数据分析摘要.csv', index=False, encoding='utf-8-sig')
    print("\n详细分析结果已保存到 '交易数据分析摘要.csv'")