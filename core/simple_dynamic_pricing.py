class SimpleDynamicPricing:
    def calculate_discount(self, current_time, shelf_time,
                           initial_inventory, remaining_inventory,
                           base_price, cost_price):
        """
        参数：
        current_time: 当前时间（小时，0-23）
        shelf_time: 上架时间（小时）
        initial_inventory: 初始库存
        remaining_inventory: 剩余库存
        base_price: 基准售价
        cost_price: 成本价

        返回：
        建议折扣率（0-1之间）
        """

        # 计算剩余时间比例
        remaining_hours = 24 - current_time
        time_ratio = remaining_hours / 24

        # 计算库存压力
        inventory_ratio = remaining_inventory / initial_inventory

        # 基础折扣计算
        base_discount = 0.95  # 基础折扣95折

        # 时间压力因子（越晚折扣越大）
        time_pressure = 1.2 * (1 - time_ratio ** 0.5)

        # 库存压力因子（库存越多折扣越大）
        inventory_pressure = 0.8 * inventory_ratio

        # 综合折扣率
        discount_rate = base_discount - time_pressure - inventory_pressure

        # 确保折扣不低于成本价
        min_discount = cost_price / base_price
        discount_rate = max(discount_rate, min_discount)

        # 折扣上限（如不低于5折）
        discount_rate = max(discount_rate, 0.5)

        return round(discount_rate, 2)