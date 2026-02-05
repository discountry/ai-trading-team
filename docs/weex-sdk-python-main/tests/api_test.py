"""WEEX API Test Suite using weex-sdk."""

import math
import time
from typing import Optional

import os

from dotenv import load_dotenv
from weex_sdk import WeexClient

load_dotenv()  # 默认读取当前目录下的 .env

# 初始化客户端
client = WeexClient(
    api_key=os.getenv("API_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    passphrase=os.getenv("ACCESS_PASSPHRASE"),
)


def adjust_price_to_step_size(price: float, step_size: float = 0.1) -> float:
    """
    调整价格以符合步长要求
    :param price: 原始价格
    :param step_size: 步长，默认0.1
    :return: 调整后的价格（向下取整到最近的步长倍数）
    """
    # 向下取整到最近的步长倍数
    adjusted_price = math.floor(price / step_size) * step_size
    # 保留适当的小数位数（根据步长确定）
    if step_size >= 1:
        decimal_places = 0
    elif step_size >= 0.1:
        decimal_places = 1
    elif step_size >= 0.01:
        decimal_places = 2
    elif step_size >= 0.001:
        decimal_places = 3
    else:
        decimal_places = 4

    return round(adjusted_price, decimal_places)


def check_balance():
    """调用账户资产查询接口"""
    print("\n=== 账户资产查询 ===")
    assets = client.account.get_assets()
    for asset in assets:
        print(f"币种: {asset.get('coinName')}")
        print(f"  可用: {asset.get('available')}")
        print(f"  冻结: {asset.get('frozen')}")
        print(f"  权益: {asset.get('equity')}")
        print(f"  未实现盈亏: {asset.get('unrealizePnl')}")
    return assets


def get_price(symbol: str):
    """
    获取标的当前行情价格等信息
    :param symbol: 交易对，例如 'cmt_btcusdt'
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 获取 {symbol} 价格 ===")
    ticker = client.market.get_ticker(symbol)
    print(f"当前价格: {ticker.get('last')}")
    print(f"最佳买价: {ticker.get('best_bid')}")
    print(f"最佳卖价: {ticker.get('best_ask')}")
    print(f"24h涨跌: {ticker.get('priceChangePercent')}%")
    return ticker


def change_leverage(symbol: str, margin_mode: int, long_leverage: str, short_leverage: Optional[str] = None):
    """
    调整杠杆倍数
    :param symbol: 交易对，例如 'cmt_btcusdt'
    :param margin_mode: 保证金模式，1：全仓，3：逐仓
    :param long_leverage: 多头杠杆，例如 '2'
    :param short_leverage: 空头杠杆，例如 '2'；在全仓模式下必须与 long_leverage 相同
    """
    print(f"\n=== 调整杠杆 {symbol} ===")
    print(f"保证金模式: {'全仓' if margin_mode == 1 else '逐仓'}")
    print(f"多头杠杆: {long_leverage}x")
    if short_leverage:
        print(f"空头杠杆: {short_leverage}x")

    result = client.account.set_leverage(
        symbol=symbol,
        margin_mode=margin_mode,
        long_leverage=long_leverage,
        short_leverage=short_leverage
    )
    print(f"结果: {result}")
    return result


def get_position(symbol: str):
    """查询单个交易对的持仓"""
    print(f"\n=== 查询持仓 {symbol} ===")
    position = client.account.get_single_position(symbol)
    print(position)
    return position


def place_order(
    symbol: str,
    client_oid: str,
    size: str,
    order_type: str,
    match_price: str,
    price: str,
    type_order: str,
    preset_take_profit_price: Optional[str] = None,
    preset_stop_loss_price: Optional[str] = None,
    margin_mode: Optional[int] = None,
):
    """
    下单
    :param symbol: 交易对，例如 'cmt_btcusdt'
    :param client_oid: 自定义订单ID（不超过40个字符）
    :param size: 订单数量（不能为零或负数）
    :param order_type: 订单类型 0: 普通, 1: 只做Maker, 2: Fill-Or-Kill, 3: Immediate Or Cancel
    :param match_price: 0: 限价, 1: 市价
    :param price: 订单价格（限价单必填，精度和步长遵循合约信息接口）
    :param type_order: 1: 开多, 2: 开空, 3: 平多, 4: 平空
    :param preset_take_profit_price: 预设止盈价格（可选）
    :param preset_stop_loss_price: 预设止损价格（可选）
    :param margin_mode: 保证金模式 1: 全仓, 3: 逐仓，默认为1（全仓）
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 下单 {symbol} ===")
    print(f"订单方向: {get_order_type_name(type_order)}")
    print(f"数量: {size}")
    print(f"价格类型: {'市价' if match_price == '1' else '限价'}")
    if match_price == '0':
        print(f"价格: {price}")

    result = client.trade.place_order(
        symbol=symbol,
        client_oid=client_oid,
        size=size,
        order_type=order_type,
        match_price=match_price,
        price=price,
        type=type_order,
        preset_take_profit_price=preset_take_profit_price,
        preset_stop_loss_price=preset_stop_loss_price,
        margin_mode=margin_mode,
    )

    print(f"订单ID: {result.get('order_id')}")
    print(f"客户端订单ID: {result.get('client_oid')}")
    return result


def cancel_order(order_id: Optional[str] = None, client_oid: Optional[str] = None):
    """
    取消订单
    :param order_id: 订单ID（orderId 或 clientOid 必须提供一个）
    :param client_oid: 客户端自定义ID（orderId 或 clientOid 必须提供一个）
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 取消订单 ===")
    if order_id:
        print(f"订单ID: {order_id}")
    if client_oid:
        print(f"客户端订单ID: {client_oid}")

    result = client.trade.cancel_order(order_id=order_id, client_oid=client_oid)
    print(f"取消结果: {result}")
    return result


def cancel_batch_orders(ids: Optional[list] = None, cids: Optional[list] = None):
    """
    批量取消订单
    :param ids: 订单ID列表（ids 或 cids 必须提供一个）
    :param cids: 客户端自定义ID列表（ids 或 cids 必须提供一个）
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 批量取消订单 ===")
    result = client.trade.cancel_batch_orders(ids=ids, cids=cids)
    print(f"批量取消结果: {result}")
    return result


def get_order_detail(order_id: str):
    """
    查询指定订单
    :param order_id: 订单ID
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 查询订单详情 {order_id} ===")
    order = client.trade.get_order_detail(order_id)
    print(f"订单状态: {order.get('status')}")
    print(f"订单类型: {get_order_type_name(order.get('type'))}")
    print(f"数量: {order.get('size')}")
    print(f"已成交数量: {order.get('filled_qty')}")
    print(f"价格: {order.get('price')}")
    print(f"平均成交价: {order.get('price_avg')}")
    print(f"手续费: {order.get('fee')}")
    print(f"总盈亏: {order.get('totalProfits')}")
    return order


def get_order_history(symbol: Optional[str] = None, page_size: Optional[int] = None, create_date: Optional[int] = None):
    """
    获取历史挂单
    :param symbol: 交易对（可选）
    :param page_size: 每页数量（可选）
    :param create_date: 创建时间，Unix毫秒时间戳（可选，必须≤90且不能为负数）
    :return: 解析后的 JSON 数据（list）
    """
    print(f"\n=== 获取历史订单 ===")
    orders = client.trade.get_order_history(symbol=symbol, page_size=page_size, create_date=create_date)
    print(f"历史订单数量: {len(orders)}")
    for order in orders[:5]:  # 只显示前5个
        print(f"  - 订单ID: {order.get('order_id')}, 状态: {order.get('status')}, 类型: {get_order_type_name(order.get('type'))}")
    return orders


def get_current_orders(
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
    page: Optional[int] = None,
):
    """
    获取当前挂单
    :param symbol: 交易对（可选）
    :param order_id: 订单ID（可选）
    :param start_time: 查询记录开始时间，Unix毫秒时间戳（可选）
    :param end_time: 查询记录结束时间，Unix毫秒时间戳（可选）
    :param limit: 限制数量，默认100，最大100（可选）
    :param page: 页码，默认0（可选）
    :return: 解析后的 JSON 数据（list）
    """
    print(f"\n=== 获取当前挂单 ===")
    orders = client.trade.get_current_orders(
        symbol=symbol,
        order_id=order_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        page=page,
    )
    print(f"当前挂单数量: {len(orders)}")
    for order in orders:
        print(f"  - 订单ID: {order.get('order_id')}, 数量: {order.get('size')}, 状态: {order.get('status')}")
    return orders


def cancel_all_orders(cancel_order_type: str, symbol: Optional[str] = None):
    """
    撤销全部订单
    :param cancel_order_type: 取消订单类型 'normal': 取消普通订单, 'plan': 取消触发/计划订单
    :param symbol: 交易对（可选，如果不提供则取消所有交易对的订单）
    :return: 解析后的 JSON 数据（list）
    """
    print(f"\n=== 撤销全部订单 ===")
    print(f"类型: {cancel_order_type}")
    if symbol:
        print(f"交易对: {symbol}")

    results = client.trade.cancel_all_orders(cancel_order_type=cancel_order_type, symbol=symbol)
    print(f"撤销结果: {results}")
    return results


def close_positions(symbol: Optional[str] = None):
    """
    一键平仓
    :param symbol: 交易对（可选，如果不提供则以市价平掉所有持仓）
    :return: 解析后的 JSON 数据（list）
    """
    print(f"\n=== 一键平仓 ===")
    if symbol:
        print(f"交易对: {symbol}")
    else:
        print("平掉所有持仓")

    results = client.trade.close_positions(symbol=symbol)
    print(f"平仓结果: {results}")
    return results


def get_order_fills(
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None,
):
    """
    获取订单成交明细
    :param symbol: 交易对名称（可选）
    :param order_id: 订单ID（可选）
    :param start_time: 开始时间戳，Unix毫秒时间戳（可选）
    :param end_time: 结束时间戳，Unix毫秒时间戳（可选）
    :param limit: 查询数量，最大100，默认100（可选）
    :return: 解析后的 JSON 数据（dict）
    """
    print(f"\n=== 获取订单成交明细 ===")
    fills = client.trade.get_order_fills(
        symbol=symbol,
        order_id=order_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )

    items = fills.get('data', []) if isinstance(fills, dict) else []
    print(f"成交记录数量: {len(items)}")
    for fill in items[:5]:  # 只显示前5条
        print(f"  - 交易ID: {fill.get('tradeId')}, 方向: {fill.get('direction')}, 成交量: {fill.get('fillSize')}")
    return fills


def get_current_price_and_place_limit_order(
    symbol: str,
    size: str,
    order_type_direction: int,
    price_offset_percent: float = 20,
    step_size: float = 0.1,
):
    """
    获取当前真实价格并挂一个不会立即成交的限价单
    :param symbol: 交易对，例如 'cmt_btcusdt'
    :param size: 订单数量
    :param order_type_direction: 订单方向 1: 开多, 2: 开空, 3: 平多, 4: 平空
    :param price_offset_percent: 价格偏移百分比，默认20%（开多时价格低于市价，开空时价格高于市价）
    :param step_size: 价格步长，默认0.1
    :return: 订单结果（dict）
    """
    # 1. 获取当前价格
    print(f"\n=== 步骤1: 获取 {symbol} 当前价格 ===")
    price_info = get_price(symbol)
    if not price_info:
        print("获取价格失败")
        return None

    current_price = float(price_info.get("last", 0))
    best_bid = float(price_info.get("best_bid", current_price))
    best_ask = float(price_info.get("best_ask", current_price))

    print(f"当前价格: {current_price}")
    print(f"最佳买价: {best_bid}")
    print(f"最佳卖价: {best_ask}")

    # 2. 计算限价单价格（确保不会立即成交）
    # 开多：价格低于最佳买价
    # 开空：价格高于最佳卖价
    if order_type_direction == 1:  # 开多
        limit_price = best_bid * (1 - price_offset_percent / 100)
        print(f"\n=== 步骤2: 挂开多限价单（价格低于市价 {price_offset_percent}%） ===")
    elif order_type_direction == 2:  # 开空
        limit_price = best_ask * (1 + price_offset_percent / 100)
        print(f"\n=== 步骤2: 挂开空限价单（价格高于市价 {price_offset_percent}%） ===")
    elif order_type_direction == 3:  # 平多
        limit_price = best_bid * (1 - price_offset_percent / 100)
        print(f"\n=== 步骤2: 挂平多限价单（价格低于市价 {price_offset_percent}%） ===")
    elif order_type_direction == 4:  # 平空
        limit_price = best_ask * (1 + price_offset_percent / 100)
        print(f"\n=== 步骤2: 挂平空限价单（价格高于市价 {price_offset_percent}%） ===")
    else:
        print(f"无效的订单方向: {order_type_direction}")
        return None

    # 3. 调整价格以符合步长要求
    limit_price_adjusted = adjust_price_to_step_size(limit_price, step_size)
    print(f"计算出的限价单价格: {limit_price}")
    print(f"调整后的限价单价格（符合步长 {step_size}）: {limit_price_adjusted}")

    # 确保价格大于0
    if limit_price_adjusted <= 0:
        print(f"错误: 调整后的价格 {limit_price_adjusted} 必须大于0")
        return None

    # 4. 生成客户端订单ID
    client_oid = f"{int(time.time() * 1000)}_{order_type_direction}"

    # 5. 下单
    print(f"\n=== 步骤3: 提交限价单 ===")
    order_result = place_order(
        symbol=symbol,
        client_oid=client_oid,
        size=size,
        order_type="0",  # 0: 普通限价单
        match_price="0",  # 0: 限价
        price=str(limit_price_adjusted),
        type_order=str(order_type_direction),
    )

    if order_result:
        print(f"订单提交成功!")
    else:
        print("订单提交失败")

    return order_result


def place_market_order_open_short_and_close(symbol: str = "cmt_btcusdt", size: str = "0.0002", wait_seconds: int = 1):
    """
    市价开空订单，等待指定时间后立即平仓并检查确认
    :param symbol: 交易对，默认 'cmt_btcusdt'
    :param size: 订单数量，默认 0.0002
    :param wait_seconds: 等待时间（秒），默认1秒
    :return: (开仓订单结果, 平仓订单结果)
    """
    print("\n" + "=" * 60)
    print("市价开空并平仓流程 - 开始")
    print("=" * 60)

    # 1. 获取当前价格
    print(f"\n【步骤1】获取 {symbol} 当前价格")
    price_info = get_price(symbol)
    if not price_info:
        print("获取价格失败")
        return None, None

    current_price = float(price_info.get("last", 0))
    print(f"当前价格: {current_price}")

    # 2. 市价开空订单
    print(f"\n【步骤2】提交市价开空订单（数量: {size}）")
    client_oid_open = f"{int(time.time() * 1000)}_open_short"

    open_order_result = place_order(
        symbol=symbol,
        client_oid=client_oid_open,
        size=size,
        order_type="0",  # 0: 普通订单
        match_price="1",  # 1: 市价
        price="0",  # 市价单时价格可以为0
        type_order="2",  # 2: 开空
    )

    if not open_order_result or not open_order_result.get('order_id'):
        print("开空订单提交失败")
        return None, None

    open_order_id = open_order_result.get('order_id')
    print(f"✓ 开空订单提交成功!")
    print(f"  订单ID: {open_order_id}")
    print(f"  客户端订单ID: {open_order_result.get('client_oid')}")

    # 3. 等待指定时间
    print(f"\n【步骤3】等待 {wait_seconds} 秒...")
    time.sleep(wait_seconds)

    # 4. 检查开仓订单状态
    print(f"\n【步骤4】检查开仓订单状态")
    open_order_detail = get_order_detail(open_order_id)

    # 5. 市价平空订单
    print(f"\n【步骤5】提交市价平空订单（数量: {size}）")
    client_oid_close = f"{int(time.time() * 1000)}_close_short"

    close_order_result = place_order(
        symbol=symbol,
        client_oid=client_oid_close,
        size=size,
        order_type="0",  # 0: 普通订单
        match_price="1",  # 1: 市价
        price="0",  # 市价单时价格可以为0
        type_order="4",  # 4: 平空
    )

    if not close_order_result or not close_order_result.get('order_id'):
        print("平空订单提交失败")
        return open_order_result, None

    close_order_id = close_order_result.get('order_id')
    print(f"✓ 平空订单提交成功!")
    print(f"  订单ID: {close_order_id}")
    print(f"  客户端订单ID: {close_order_result.get('client_oid')}")

    # 6. 等待订单处理
    print(f"\n【步骤6】等待订单处理...")
    time.sleep(2)

    # 7. 检查确认两个订单状态
    print(f"\n【步骤7】检查确认订单状态")

    # 检查开仓订单
    print(f"\n开仓订单（ID: {open_order_id}）状态:")
    open_order_detail_final = get_order_detail(open_order_id)

    # 检查平仓订单
    print(f"\n平仓订单（ID: {close_order_id}）状态:")
    close_order_detail_final = get_order_detail(close_order_id)

    # 8. 获取成交明细
    print(f"\n【步骤8】获取成交明细")
    get_order_fills(symbol=symbol, order_id=open_order_id, limit=10)
    get_order_fills(symbol=symbol, order_id=close_order_id, limit=10)

    print("\n" + "=" * 60)
    print("市价开空并平仓流程 - 完成")
    print("=" * 60)

    return open_order_result, close_order_result


def complete_api_test(symbol: str = "cmt_btcusdt"):
    """
    完成所有 API 测试任务（按照 WEEX Hackathon 指南）
    :param symbol: 交易对，默认 'cmt_btcusdt'
    """
    print("\n" + "=" * 60)
    print("WEEX API 测试流程 - 开始")
    print("=" * 60)

    # 步骤3: 检查账户余额
    print("\n【步骤3】检查账户余额")
    check_balance()

    # 步骤4: 设置杠杆
    print("\n【步骤4】设置杠杆（全仓模式，2倍杠杆）")
    change_leverage(symbol, 1, "2")

    # 步骤5: 获取资产价格
    print("\n【步骤5】获取资产价格")
    price_info = get_price(symbol)
    if price_info:
        print(f"当前价格: {price_info.get('last')}")
        print(f"24小时涨跌幅: {price_info.get('priceChangePercent')}%")

    # 步骤6-8: 获取价格并挂限价单（不会立即成交）
    print("\n【步骤6-8】获取当前价格并挂限价单")
    order_result = get_current_price_and_place_limit_order(
        symbol=symbol,
        size="0.01",
        order_type_direction=1,  # 开多
        price_offset_percent=20  # 价格低于市价20%，确保不会立即成交
    )

    if order_result and order_result.get('order_id'):
        order_id = order_result.get('order_id')

        # 步骤9: 获取当前挂单信息
        print("\n【步骤9】获取当前挂单信息")
        get_current_orders(symbol=symbol)

        # 步骤10: 获取历史订单记录
        print("\n【步骤10】获取历史订单记录")
        create_date = int(time.time() * 1000)
        get_order_history(symbol=symbol, page_size=10, create_date=create_date)

        # 步骤11: 获取成交明细
        print("\n【步骤11】获取订单成交明细")
        get_order_fills(symbol=symbol, order_id=order_id, limit=10)

        # 取消所有挂单并验证
        print("\n" + "=" * 60)
        print("【清理步骤】取消所有挂单并验证")
        print("=" * 60)

        # 取消所有普通订单
        print("\n取消所有普通订单...")
        cancel_result = cancel_all_orders(cancel_order_type="normal", symbol=symbol)
        if cancel_result:
            print(f"取消结果: {cancel_result}")

        # 等待一下确保订单状态更新
        time.sleep(1)

        # 验证是否还有未撤销的订单
        print("\n验证所有订单是否已撤销...")
        remaining_orders = get_current_orders(symbol=symbol)

        if remaining_orders and len(remaining_orders) > 0:
            print(f"警告: 仍有 {len(remaining_orders)} 个未撤销的订单")
            for order in remaining_orders:
                print(f"  - 订单ID: {order.get('order_id')}, 状态: {order.get('status')}")

            # 尝试再次取消
            print("\n尝试再次取消剩余订单...")
            cancel_result2 = cancel_all_orders(cancel_order_type="normal", symbol=symbol)
            time.sleep(1)

            # 再次验证
            remaining_orders2 = get_current_orders(symbol=symbol)
            if remaining_orders2 and len(remaining_orders2) > 0:
                print(f"仍有 {len(remaining_orders2)} 个订单未能撤销")
            else:
                print("✓ 所有订单已成功撤销")
        else:
            print("✓ 所有订单已成功撤销")

        print("\n" + "=" * 60)
        print("WEEX API 测试流程 - 完成")
        print("=" * 60)
    else:
        print("\n订单提交失败，无法继续后续步骤")

        # 即使订单提交失败，也尝试清理可能存在的挂单
        print("\n清理可能存在的挂单...")
        cancel_result = cancel_all_orders(cancel_order_type="normal", symbol=symbol)
        if cancel_result:
            print(f"取消结果: {cancel_result}")


def get_order_type_name(type_code: Optional[str]) -> str:
    """获取订单类型名称"""
    type_map = {
        "1": "开多",
        "2": "开空",
        "3": "平多",
        "4": "平空",
    }
    return type_map.get(str(type_code), "未知")


if __name__ == '__main__':

    # 市价开空并平仓流程（0.0002，1秒后平仓）
    place_market_order_open_short_and_close("cmt_btcusdt", "0.0002", 1)
