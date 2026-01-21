import pandas as pd
import numpy as np


def analyze_by_category(df):
    """
    按小类编码进行统计分析

    参数:
    df: pandas DataFrame，包含交易数据

    返回:
    DataFrame: 包含每个小类编码的统计结果
    dict: 包含小类编码的汇总指标
    """
    # 确保数据中有小类编码列
    if '小类编码' not in df.columns:
        print("警告: 数据中没有'小类编码'列")
        return pd.DataFrame(), {}

    # 确保数值列是数值类型
    numeric_cols = ['销售金额', '折扣金额', '销售数量', '销售净额']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算每个小类编码的统计指标
    category_stats = df.groupby('小类编码').agg({
        '销售金额': 'sum',  # 总销售额
        '折扣金额': 'sum',  # 总折扣金额
        '销售数量': 'sum',  # 总销量
        '流水单号': 'nunique',  # 订单数量
        '商品编码': 'nunique'  # 商品种类数
    }).reset_index()

    # 重命名列
    category_stats.columns = ['小类编码', '销售额', '折扣金额', '销量', '订单数', '商品种类数']

    # 计算衍生指标
    category_stats['实际销售额'] = category_stats['销售额'] - category_stats['折扣金额']
    category_stats['折扣率'] = np.where(
        category_stats['销售额'] > 0,
        category_stats['折扣金额'] / category_stats['销售额'] * 100,
        0
    )
    category_stats['平均单价'] = np.where(
        category_stats['销量'] > 0,
        category_stats['实际销售额'] / category_stats['销量'],
        0
    )
    category_stats['平均每单金额'] = np.where(
        category_stats['订单数'] > 0,
        category_stats['实际销售额'] / category_stats['订单数'],
        0
    )
    category_stats['商品平均销量'] = np.where(
        category_stats['商品种类数'] > 0,
        category_stats['销量'] / category_stats['商品种类数'],
        0
    )

    # 按销售额排序
    category_stats = category_stats.sort_values('销售额', ascending=False)

    # 计算占比
    total_sales = category_stats['销售额'].sum()
    total_actual_sales = category_stats['实际销售额'].sum()
    total_quantity = category_stats['销量'].sum()
    total_orders = category_stats['订单数'].sum()

    category_stats['销售额占比'] = (category_stats['销售额'] / total_sales * 100) if total_sales > 0 else 0
    category_stats['实际销售额占比'] = (
                category_stats['实际销售额'] / total_actual_sales * 100) if total_actual_sales > 0 else 0
    category_stats['销量占比'] = (category_stats['销量'] / total_quantity * 100) if total_quantity > 0 else 0
    category_stats['订单数占比'] = (category_stats['订单数'] / total_orders * 100) if total_orders > 0 else 0

    # 计算小类编码的汇总指标
    category_summary = {
        '小类编码总数': len(category_stats),
        '总销售额': round(total_sales, 2),
        '总实际销售额': round(total_actual_sales, 2),
        '总销量': int(total_quantity),
        '总订单数': int(total_orders),
        '平均折扣率': round(category_stats['折扣率'].mean(), 2),
        '前5大小类销售额占比': round(category_stats.head(5)['销售额占比'].sum(), 2),
        '前5大小类实际销售额占比': round(category_stats.head(5)['实际销售额占比'].sum(), 2),
        '前5大小类销量占比': round(category_stats.head(5)['销量占比'].sum(), 2),
        '前5大小类订单数占比': round(category_stats.head(5)['订单数占比'].sum(), 2),
    }

    # 获取销售额前10的小类编码
    top_10_categories = category_stats.head(10)[
        ['小类编码', '销售额', '实际销售额', '折扣率', '销量', '订单数', '销售额占比']]
    category_summary['销售额前10小类'] = top_10_categories.to_dict('records')

    # 获取折扣率最高的小类（排除零销售额）
    high_discount_categories = category_stats[category_stats['销售额'] > 0].sort_values('折扣率', ascending=False).head(
        10)
    category_summary['折扣率最高前10小类'] = high_discount_categories[
        ['小类编码', '折扣率', '销售额', '实际销售额']].to_dict('records')

    # 获取商品种类最多的小类
    high_variety_categories = category_stats.sort_values('商品种类数', ascending=False).head(10)
    category_summary['商品种类数前10小类'] = high_variety_categories[
        ['小类编码', '商品种类数', '销售额', '销量']].to_dict('records')

    # 格式化数字，保留2位小数
    float_columns = ['销售额', '折扣金额', '实际销售额', '折扣率', '平均单价',
                     '平均每单金额', '商品平均销量', '销售额占比', '实际销售额占比',
                     '销量占比', '订单数占比']

    for col in float_columns:
        if col in category_stats.columns:
            category_stats[col] = category_stats[col].round(2)

    return category_stats, category_summary


def print_category_analysis(category_stats, category_summary):
    """打印小类编码分析结果"""
    if category_stats.empty:
        print("没有小类编码数据可分析")
        return

    print("=" * 80)
    print("小类编码统计分析报告")
    print("=" * 80)

    # 打印汇总指标
    print("\n汇总指标:")
    print("-" * 80)
    print(f"小类编码总数: {category_summary['小类编码总数']}")
    print(f"总销售额: ¥{category_summary['总销售额']:,.2f}")
    print(f"总实际销售额: ¥{category_summary['总实际销售额']:,.2f}")
    print(f"总销量: {category_summary['总销量']:,}")
    print(f"总订单数: {category_summary['总订单数']:,}")
    print(f"平均折扣率: {category_summary['平均折扣率']:.2f}%")
    print(f"前5大小类销售额占比: {category_summary['前5大小类销售额占比']:.2f}%")
    print(f"前5大小类实际销售额占比: {category_summary['前5大小类实际销售额占比']:.2f}%")
    print(f"前5大小类销量占比: {category_summary['前5大小类销量占比']:.2f}%")
    print(f"前5大小类订单数占比: {category_summary['前5大小类订单数占比']:.2f}%")

    # 打印销售额前10的小类
    print("\n销售额前10的小类编码:")
    print("-" * 80)
    print(
        f"{'排名':<5} {'小类编码':<15} {'销售额':<15} {'实际销售额':<15} {'折扣率':<10} {'销量':<10} {'订单数':<10} {'占比':<10}")
    print("-" * 80)
    for i, (_, row) in enumerate(category_stats.head(10).iterrows(), 1):
        print(
            f"{i:<5} {row['小类编码']:<15} ¥{row['销售额']:<14,.2f} ¥{row['实际销售额']:<14,.2f} {row['折扣率']:<9.2f}% {row['销量']:<10} {row['订单数']:<10} {row['销售额占比']:<9.2f}%")

    # 打印折扣率最高的10个小类（排除零销售额）
    high_discount = category_stats[category_stats['销售额'] > 0].sort_values('折扣率', ascending=False).head(10)
    if not high_discount.empty:
        print("\n折扣率最高的10个小类编码:")
        print("-" * 80)
        print(f"{'排名':<5} {'小类编码':<15} {'折扣率':<15} {'销售额':<15} {'实际销售额':<15}")
        print("-" * 80)
        for i, (_, row) in enumerate(high_discount.iterrows(), 1):
            print(
                f"{i:<5} {row['小类编码']:<15} {row['折扣率']:<14.2f}% ¥{row['销售额']:<14,.2f} ¥{row['实际销售额']:<14,.2f}")

    # 打印商品种类最多的10个小类
    print("\n商品种类最多的10个小类编码:")
    print("-" * 80)
    print(f"{'排名':<5} {'小类编码':<15} {'商品种类数':<15} {'销售额':<15} {'销量':<15}")
    print("-" * 80)
    high_variety = category_stats.sort_values('商品种类数', ascending=False).head(10)
    for i, (_, row) in enumerate(high_variety.iterrows(), 1):
        print(f"{i:<5} {row['小类编码']:<15} {row['商品种类数']:<15} ¥{row['销售额']:<14,.2f} {row['销量']:<15}")


def analyze_transaction_data_with_category(df):
    """
    综合分析方法，包含整体统计和小类编码分析

    参数:
    df: pandas DataFrame，包含交易数据

    返回:
    dict: 包含整体分析结果和小类编码分析结果
    """

    # 原有整体分析方法
    def original_analysis(df):
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

        # 计算关键指标
        total_sales = df['销售金额'].sum()
        total_discount = df['折扣金额'].sum()
        total_net_sales = total_sales - total_discount

        if '流水单号' in df.columns:
            daily_orders = df.groupby('日期')['流水单号'].nunique()
            total_orders = df['流水单号'].nunique()
            days_with_orders = len(daily_orders)
            avg_daily_orders = total_orders / days_with_orders if days_with_orders > 0 else 0
        else:
            total_orders = 0
            days_with_orders = 0
            avg_daily_orders = 0

        avg_discount_rate = (total_discount / total_sales * 100) if total_sales > 0 else 0

        return {
            '总销售额': round(total_sales, 2),
            '总折扣金额': round(total_discount, 2),
            '实际销售额': round(total_net_sales, 2),
            '总订单数量': total_orders,
            '日均订单数量': round(avg_daily_orders, 2),
            '平均折扣率': round(avg_discount_rate, 2)
        }

    # 执行整体分析
    overall_results = original_analysis(df)

    # 执行小类编码分析
    category_stats, category_summary = analyze_by_category(df)

    # 合并结果
    combined_results = {
        '整体分析': overall_results,
        '小类编码分析': category_summary,
        '小类编码详细数据': category_stats
    }

    return combined_results


def print_comprehensive_report(results):
    """打印综合报告"""
    print("=" * 100)
    print("综合交易数据分析报告")
    print("=" * 100)

    # 打印整体分析结果
    print("\n整体分析结果:")
    print("-" * 100)
    overall = results['整体分析']
    print(f"总销售额: ¥{overall['总销售额']:,.2f}")
    print(f"总折扣金额: ¥{overall['总折扣金额']:,.2f}")
    print(f"实际销售额: ¥{overall['实际销售额']:,.2f}")
    print(f"总订单数量: {overall['总订单数量']:,}")
    print(f"日均订单数量: {overall['日均订单数量']:.2f}")
    print(f"平均折扣率: {overall['平均折扣率']:.2f}%")

    # 打印小类编码分析摘要
    if results['小类编码分析']:
        print("\n小类编码分析摘要:")
        print("-" * 100)
        category_summary = results['小类编码分析']
        print(f"小类编码总数: {category_summary['小类编码总数']}")
        print(f"前5大小类销售额占比: {category_summary['前5大小类销售额占比']:.2f}%")
        print(f"前5大小类实际销售额占比: {category_summary['前5大小类实际销售额占比']:.2f}%")
        print(f"前5大小类销量占比: {category_summary['前5大小类销量占比']:.2f}%")
        print(f"前5大小类订单数占比: {category_summary['前5大小类订单数占比']:.2f}%")

    # 打印销售额前10的小类
    if '销售额前10小类' in results['小类编码分析']:
        print("\n销售额前10的小类编码:")
        print("-" * 100)
        print(f"{'排名':<5} {'小类编码':<15} {'销售额':<20} {'实际销售额':<20} {'折扣率':<15}")
        print("-" * 100)
        for i, item in enumerate(results['小类编码分析']['销售额前10小类'], 1):
            print(
                f"{i:<5} {item['小类编码']:<15} ¥{item['销售额']:<19,.2f} ¥{item['实际销售额']:<19,.2f} {item['折扣率']:<14.2f}%")


# 使用示例
if __name__ == "__main__":
    # 读取CSV文件
    # 注意：根据您的文件编码和分隔符调整参数
    df = pd.read_csv('K5.交易流水明细表2026-01-13_9_49_12_xian.csv', encoding='utf-8')

    # 方法1: 只进行小类编码分析
    print("方法1: 小类编码分析")
    print("=" * 80)
    category_stats, category_summary = analyze_by_category(df)
    print_category_analysis(category_stats, category_summary)

    # 保存小类编码分析结果到CSV
    if not category_stats.empty:
        category_stats.to_csv('小类编码分析结果.csv', index=False, encoding='utf-8-sig')
        print(f"\n小类编码分析结果已保存到 '小类编码分析结果.csv'")

    # 方法2: 综合分析方法
    print("\n\n方法2: 综合分析方法")
    results = analyze_transaction_data_with_category(df)
    print_comprehensive_report(results)

    # 保存综合报告到Excel
    with pd.ExcelWriter('综合交易分析报告.xlsx', engine='openpyxl') as writer:
        # 保存整体分析结果
        overall_df = pd.DataFrame([results['整体分析']])
        overall_df.to_excel(writer, sheet_name='整体分析', index=False)

        # 保存小类编码详细数据
        if not results['小类编码详细数据'].empty:
            results['小类编码详细数据'].to_excel(writer, sheet_name='小类编码分析', index=False)

        # 保存小类编码汇总指标
        if results['小类编码分析']:
            # 排除销售额前10小类等复杂结构
            summary_data = {k: v for k, v in results['小类编码分析'].items()
                            if not isinstance(v, list)}
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='小类编码汇总', index=False)

    print(f"\n综合报告已保存到 '综合交易分析报告.xlsx'")