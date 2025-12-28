import pandas as pd

# 加载销售数据
hsr_df = pd.read_csv('data/historical_transactions.csv', encoding='utf-8', parse_dates=['日期', '交易时间'],
                     dtype={"商品编码": str, "门店编码": str, "流水单号": str, "会员id": str})
result = hsr_df.groupby(['门店编码', '商品编码']).size().reset_index(name='出现次数')
result = result.sort_values('出现次数', ascending=False)
# 获取总体前10名（不考虑门店）
top_10_overall = hsr_df['商品编码'].value_counts().head(10)
print("方法1 - 总体前10个商品编码（不考虑门店）:")
print(top_10_overall)
# 进口蓝莓（盒）125克/盒
# 润家蓝莓125g