# 策略运行逻辑和生命周期完整文档

> 最后更新: 2026-01-11

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TradingBot (main.py)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ DataManager  │───▶│   DataPool   │◀───│ StrategyOrchestrator │  │
│  │  (Binance)   │    │ (共享数据池)  │    │    (策略编排器)       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ RiskMonitor  │    │ SignalQueue  │    │  StrategyStateMachine│  │
│  │  (风控监控)   │    │  (信号队列)   │    │    (状态机)          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Executor   │◀───│    Agent     │◀───│      Factors         │  │
│  │    (WEEX)    │    │ (MiniMax AI) │    │   (5个单因子策略)     │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心组件说明

| 组件 | 文件 | 职责 |
|------|------|------|
| TradingBot | `main.py` | 主入口，协调所有组件 |
| DataPool | `core/data_pool.py` | 线程安全的共享数据存储 |
| SignalQueue | `core/signal_queue.py` | 时间戳信号队列，防止重复 |
| StrategyOrchestrator | `strategy/orchestrator.py` | 多因子策略聚合与信号生成 |
| StrategyStateMachine | `strategy/state_machine.py` | 交易生命周期状态管理 |
| LangChainTradingAgent | `agent/trader.py` | AI决策引擎 |
| RiskMonitor | `risk/monitor.py` | 独立风控监控 |
| WEEXExecutor | `execution/weex/executor.py` | 交易执行 |

---

## 二、AI模型配置

### 当前使用的模型

- **模型**: MiniMax-M2.1
- **服务商**: MiniMax (兼容 Anthropic API)
- **基础URL**: `https://api.minimax.io/anthropic`

### 环境变量配置

```bash
export ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
export ANTHROPIC_API_KEY=${YOUR_API_KEY}
```

### 代码配置位置

```python
# src/ai_trading_team/agent/trader.py
from langchain_anthropic import ChatAnthropic

ChatAnthropic(
    model="MiniMax-M2.1",
    anthropic_api_key=config.api.anthropic_api_key,
    anthropic_api_url=config.api.anthropic_base_url,
    max_tokens=2048,
)
```

---

## 三、多因子策略系统

### 因子权重配置 (总和 = 1.0)

| 因子 | 权重 | 策略类 | 描述 |
|------|------|--------|------|
| **MA Position** | 30% | `MAPositionStrategy` | 价格相对1H MA60的位置 |
| **RSI** | 20% | `RSIOversoldStrategy` | RSI超买超卖 |
| **Volatility** | 20% | `VolatilityStrategy` | 波动率过滤器 |
| **Funding Rate** | 15% | `FundingRateStrategy` | 资金费率（逆向） |
| **Long/Short Ratio** | 15% | `LongShortRatioStrategy` | 多空比（逆向） |

### 各因子评分逻辑

#### 1. MA Position (权重 30%) - 最重要的趋势指标

```python
# 文件: strategy/factors/ma_position.py
价格 > 1H MA60  → bullish_score = +1.0 (做多倾向)
价格 < 1H MA60  → bullish_score = -1.0 (做空倾向)
```

**信号类型**:
- `PRICE_ABOVE_MA`: 价格上穿MA → 看多
- `PRICE_BELOW_MA`: 价格下穿MA → 看空

#### 2. RSI (权重 20%)

```python
# 文件: strategy/factors/rsi_oversold.py
RSI ≤ 30 (超卖)   → bullish_score = +0.8 (做多机会)
RSI ≥ 70 (超买)   → bullish_score = -0.8 (做空机会)
RSI < 50          → bullish_score = +0.2
RSI > 50          → bullish_score = -0.2
```

**信号类型**:
- `RSI_OVERSOLD`: RSI ≤ 30 → 做多机会
- `RSI_OVERBOUGHT`: RSI ≥ 70 → 做空机会

#### 3. Funding Rate (权重 15%) - 逆向思维

```python
# 文件: strategy/factors/funding_rate.py
资金费率 > 0 (多头支付) → bullish_score = -0.6 (多头过热，做空)
资金费率 < 0 (空头支付) → bullish_score = +0.6 (空头过热，做多)

# 极端阈值
|funding| > 0.1%  → 强信号
|funding| > 0.01% → 普通信号
```

**信号类型**:
- `FUNDING_POSITIVE`: 正资金费率 → 看空
- `FUNDING_NEGATIVE`: 负资金费率 → 看多
- `FUNDING_EXTREME_POSITIVE`: 极端正资金费率 → 强烈看空
- `FUNDING_EXTREME_NEGATIVE`: 极端负资金费率 → 强烈看多

#### 4. Long/Short Ratio (权重 15%) - 逆向思维

```python
# 文件: strategy/factors/long_short_ratio.py
空头人数占比高 → bullish_score = +0.5 (逆向做多)
多头人数占比高 → bullish_score = -0.5 (逆向做空)
```

**信号类型**:
- `LS_RATIO_LONG_DOMINANT`: 多头主导 → 逆向看空
- `LS_RATIO_SHORT_DOMINANT`: 空头主导 → 逆向看多

#### 5. Volatility (权重 20%) - 过滤器

```python
# 文件: strategy/factors/volatility.py
低波动率 → 不交易 (volatility_ok = False)
正常波动 → 可以交易
高波动率 + OI增加 + 价格窄幅震荡 → 预示大波动即将到来
```

**作用**:
- 不生成方向性信号
- 作为交易过滤器，低波动时阻止交易

### 综合评分计算

```python
# 文件: strategy/orchestrator.py

# 1. 计算各因子加权得分
total_score = Σ (factor.bullish_score × factor.weight)

# 2. 应用市场偏向调整 (当前默认 BEARISH)
bias_adjustment = {
    MarketBias.STRONGLY_BEARISH: -0.2,
    MarketBias.BEARISH: -0.1,        # ← 当前使用
    MarketBias.NEUTRAL: 0.0,
    MarketBias.BULLISH: +0.1,
    MarketBias.STRONGLY_BULLISH: +0.2,
}
total_score += bias_adjustment

# 3. 限制在 [-1, 1] 范围
total_score = max(-1.0, min(1.0, total_score))
```

### 信号强度分类

| 综合得分 | 强度 | 行为 |
|----------|------|------|
| `\|score\| ≥ 0.6` | STRONG | 发送信号给Agent |
| `\|score\| ≥ 0.3` | MODERATE | 发送信号给Agent |
| `\|score\| ≥ 0.1` | WEAK | 不发送 |
| 因子冲突 (≥2看多 且 ≥2看空) | CONFLICTING | 不发送 |

### 信号生成条件

```python
# 必须同时满足:
1. volatility_ok == True (波动率正常)
2. strength in (STRONG, MODERATE) (信号强度足够)
3. state == IDLE (当前空仓)
```

---

## 四、状态机 (State Machine)

### 状态定义

```
┌─────────┐
│  IDLE   │ ──Entry Signal──▶ ┌───────────┐
│ (空仓)  │                   │ ANALYZING │
└─────────┘                   │ (分析中)  │
     ▲                        └───────────┘
     │                              │
     │                        Context Ready
     │                              ▼
     │                        ┌───────────────┐
     │ Cooldown              │ WAITING_ENTRY │
     │ Expired               │ (等待开仓决策) │
     │                        └───────────────┘
     │                              │
┌─────────┐                   Agent Open / Order Filled
│COOLDOWN │                         │
│ (冷却)  │                         ▼
└─────────┘                   ┌─────────────┐
     ▲                        │ IN_POSITION │◀──────────┐
     │                        │  (持仓中)   │           │
Position Closed               └─────────────┘           │
     │                              │                   │
     │              ┌───────────────┼───────────────┐   │
     │              │               │               │   │
     │         Profit 10%    25% Loss         Agent Hold
     │              ▼               ▼               │   │
     │        ┌───────────┐  ┌─────────────┐       │   │
     │        │  PROFIT   │  │RISK_OVERRIDE│       │   │
     └────────│  SIGNAL   │  │ (强制平仓)  │───────┘   │
              │(盈利信号) │  └─────────────┘           │
              └───────────┘                            │
                    │                                  │
              Agent Close                              │
                    ▼                                  │
              ┌────────────┐                           │
              │WAITING_EXIT│───────Timeout────────────┘
              │(等待平仓)  │
              └────────────┘
```

### 状态说明

| 状态 | 值 | 说明 |
|------|------|------|
| IDLE | `idle` | 空仓，等待入场信号 |
| ANALYZING | `analyzing` | 收到信号，正在收集上下文 |
| WAITING_ENTRY | `waiting_entry` | 等待Agent开仓决策 |
| IN_POSITION | `in_position` | 持仓中，监控盈亏 |
| PROFIT_SIGNAL | `profit_signal` | 盈利阈值触发，询问Agent |
| WAITING_EXIT | `waiting_exit` | 等待Agent平仓决策 |
| RISK_OVERRIDE | `risk_override` | 风控触发，强制平仓中 |
| COOLDOWN | `cooldown` | 交易后冷却期 |
| ERROR | `error` | 错误状态，需要手动干预 |

### 状态转换规则

| 当前状态 | 触发事件 | 目标状态 | 超时 |
|----------|----------|----------|------|
| IDLE | ENTRY_SIGNAL | ANALYZING | - |
| ANALYZING | CONTEXT_READY | WAITING_ENTRY | 30s → IDLE |
| WAITING_ENTRY | AGENT_OPEN | IN_POSITION | 60s → IDLE |
| WAITING_ENTRY | AGENT_OBSERVE | IDLE | - |
| WAITING_ENTRY | ORDER_FAILED | IDLE | - |
| IN_POSITION | PROFIT_THRESHOLD | PROFIT_SIGNAL | - |
| IN_POSITION | RISK_TRIGGERED | RISK_OVERRIDE | - |
| PROFIT_SIGNAL | AGENT_CLOSE | WAITING_EXIT | 60s → IN_POSITION |
| PROFIT_SIGNAL | AGENT_HOLD | IN_POSITION | - |
| WAITING_EXIT | POSITION_CLOSED | COOLDOWN | 60s → IN_POSITION |
| RISK_OVERRIDE | POSITION_CLOSED | COOLDOWN | - |
| COOLDOWN | COOLDOWN_EXPIRED | IDLE | 60s自动 |

### PositionContext 数据结构

```python
@dataclass
class PositionContext:
    symbol: str = ""
    side: Side | None = None
    entry_price: Decimal = Decimal("0")
    size: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")
    leverage: int = 1
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_percent: Decimal = Decimal("0")
    entry_time: datetime | None = None
    highest_pnl_percent: Decimal = Decimal("0")  # 用于追踪止损
    last_profit_signal_threshold: Decimal = Decimal("0")  # 上次触发的10%阈值
```

---

## 五、风控规则系统

### 规则列表 (按优先级排序)

| 规则 | 优先级 | 触发条件 | 行为 |
|------|--------|----------|------|
| **ForceStopLossRule** | 100 | 亏损 ≥ 25% margin | 立即强制平仓，无需Agent确认 |
| **TrailingStopRule** | 80 | 利润从最高点回撤超过trail距离 | 立即强制平仓 |
| **DynamicTakeProfitRule** | 50 | 利润达到10%, 20%, 30%... | 请求Agent设置止损价格 |

### 1. ForceStopLossRule (强制止损)

```python
# 文件: risk/rules.py

触发条件: unrealized_pnl / margin ≤ -25%

行为:
  - 立即市价平仓
  - 不经过Agent确认
  - 优先级最高 (100)
  - 状态转换: IN_POSITION → RISK_OVERRIDE → COOLDOWN
```

### 2. TrailingStopRule (追踪止损)

```python
# 文件: risk/rules.py

配置:
  - activation_profit_percent: 15%  # 激活阈值
  - trail_distance_percent: 7%      # 追踪距离

触发条件:
  1. 利润曾达到 ≥ 15% of margin (激活)
  2. 利润从最高点回撤超过 7%

示例:
  - 最高利润: 25%
  - 止损触发点: 25% - 7% = 18%
  - 当利润跌到 ≤ 18% 时触发

行为: 立即平仓保护利润，优先级 80
```

### 3. DynamicTakeProfitRule (动态止盈信号)

```python
# 文件: risk/rules.py

触发机制:
  - 每10%增量触发一次: 10%, 20%, 30%, 40%, ...
  - 使用 _last_signaled_threshold 跟踪已触发的阈值

算法:
  1. 计算当前利润级别: (pnl_percent // 10) * 10
  2. 如果 current_level > last_signaled_threshold:
     - 更新 last_signaled_threshold = current_level
     - 返回 move_stop_loss 动作
     - 携带 profit_data 给 Agent

行为流程:
  1. 触发信号
  2. Agent收到 PROFIT_SIGNAL_PROMPT
  3. Agent根据市场情况决定止损价格
  4. 系统取消现有止损单
  5. 系统设置新的止损单
```

### 风控检查频率

```python
# main.py
risk_check_interval = 1  # 每1秒检查一次风控
strategy_check_interval = 5  # 每5秒评估一次策略
```

---

## 六、Agent决策流程

### Agent组件

- **类**: `LangChainTradingAgent`
- **模型**: MiniMax-M2.1
- **框架**: LangChain (langchain-anthropic)
- **最大Token**: 2048

### 两种决策场景

#### 场景1: 开仓决策 (process_signal)

**输入数据**:

```python
# 信号数据
signal:
  - signal_type: STRONG_BULLISH / STRONG_BEARISH
  - data:
      - composite_score: float  # 综合得分 -1 到 1
      - strength: str  # strong / moderate
      - suggested_side: str  # long / short
      - market_bias: str  # bearish / bullish / neutral
      - volatility_ok: bool
      - factor_analysis: list  # 各因子详情

# 市场快照
snapshot:
  - ticker: 当前价格、买卖价、24h涨跌
  - klines: K线数据 (最近5根)
  - orderbook: 订单簿 (买3卖3)
  - indicators: 技术指标 (RSI, MACD, BB, ATR, MA)
  - funding_rate: 资金费率
  - long_short_ratio: 多空比
  - position: 当前仓位信息
  - orders: 当前挂单
  - account: 账户余额、可用、保证金
  - recent_operations: 最近10次操作记录
```

**输出格式**:

```json
{
    "action": "open|close|add|reduce|cancel|observe",
    "symbol": "cmt_dogeusdt",
    "side": "long|short",
    "size": 100,
    "price": null,
    "order_type": "market|limit",
    "reason": "详细分析理由..."
}
```

#### 场景2: 盈利信号决策 (process_profit_signal)

**输入数据**:

```python
profit_data:
  - current_pnl_percent: float  # 当前盈利百分比
  - threshold_level: int  # 触发阈值: 10, 20, 30...
  - highest_pnl_percent: float  # 历史最高盈利
  - position_side: str  # long / short
  - entry_price: float  # 开仓价格

snapshot:
  - position: 当前仓位详情
  - market_summary: 当前价格、24h变化
  - indicators: 技术指标
```

**输出格式**:

```json
{
    "action": "move_stop_loss",
    "symbol": "cmt_dogeusdt",
    "stop_loss_price": 0.1234,
    "reason": "技术分析理由..."
}
```

### Agent可执行动作

| 动作 | 说明 | 必填字段 |
|------|------|----------|
| `open` | 开仓 | side, size |
| `close` | 平仓 | side |
| `add` | 加仓 | side, size |
| `reduce` | 减仓 | side, size |
| `cancel` | 取消挂单 | - |
| `observe` | 观望不操作 | - |
| `move_stop_loss` | 设置/移动止损 | stop_loss_price |

### 止损单执行逻辑

```python
# main.py: _execute_move_stop_loss()

async def _execute_move_stop_loss(decision):
    stop_loss_price = decision.command.stop_loss_price

    # 1. 获取当前仓位
    position = await executor.get_position(symbol)

    # 2. 取消现有止损单
    await executor.cancel_all_orders(symbol)

    # 3. 确定止损单方向
    # LONG仓位: 止损卖出 (SHORT side)
    # SHORT仓位: 止损买入 (LONG side)
    stop_side = Side.SHORT if position.side == Side.LONG else Side.LONG

    # 4. 下止损单
    order = await executor.place_order(
        symbol=symbol,
        side=stop_side,
        order_type=OrderType.STOP_MARKET,
        size=position.size,
        price=stop_loss_price,
        action="close",
    )
```

---

## 七、完整运行生命周期

### 启动阶段

```
1. 加载配置 (Config.from_env())
   ├── API密钥 (Binance, WEEX, Anthropic)
   ├── 交易参数 (杠杆、交易对)
   └── 风控参数 (止损比例等)

2. 初始化组件
   ├── DataPool (共享数据存储)
   ├── SignalQueue (信号队列)
   ├── BinanceDataManager (数据采集)
   ├── StrategyOrchestrator (策略编排)
   │   ├── MAPositionStrategy
   │   ├── RSIOversoldStrategy
   │   ├── FundingRateStrategy
   │   ├── LongShortRatioStrategy
   │   └── VolatilityStrategy
   ├── LangChainTradingAgent (AI决策)
   ├── WEEXExecutor (交易执行)
   └── RiskMonitor (风控监控)
       ├── ForceStopLossRule (25%)
       ├── DynamicTakeProfitRule (10%增量)
       └── TrailingStopRule (15%激活, 7%追踪)

3. 连接WEEX交易所
4. 设置杠杆倍数
5. 获取历史K线数据 (100根1H K线用于MA60)
6. 启动WebSocket数据流 (1m K线, ticker, orderbook)
7. 启动市场指标循环 (30秒更新资金费率、多空比、OI)
8. 进入主循环
```

### 主循环 (每100ms)

```python
while running:
    current_time = loop.time()

    # 1. 风控检查 (每1秒) - 最高优先级
    if current_time - last_risk_check >= 1:
        last_risk_check = current_time

        if orchestrator.has_position:
            action = await risk_monitor.evaluate()

            if action:
                if action.priority >= 80:
                    # 强制平仓 (ForceStopLoss, TrailingStop)
                    await executor.close_position()
                    orchestrator.force_risk_close()
                    orchestrator.on_position_closed()

                elif action.action_type == "move_stop_loss":
                    # 盈利信号 - 请求Agent设置止损
                    await handle_profit_signal(action)

    # 2. 策略评估 (每5秒)
    if current_time - last_strategy_check >= 5:
        last_strategy_check = current_time

        # 同步仓位和账户数据
        position = await executor.get_position()
        account = await executor.get_account()
        data_pool.update_position(position)
        data_pool.update_account(account)

        # 评估多因子策略
        orchestrator.evaluate()

    # 3. 处理信号队列
    signal = signal_queue.pop()
    if signal:
        await process_signal(signal)

    await asyncio.sleep(0.1)
```

### 完整交易周期示例

```
时间线:
────────────────────────────────────────────────────────────────────▶

T+0s: 策略评估
├── MA60: 价格在上方 → bullish (+1.0 × 0.30 = +0.30)
├── RSI: 45 → neutral (+0.2 × 0.20 = +0.04)
├── Funding: -0.02% → bullish (+0.6 × 0.15 = +0.09)
├── L/S Ratio: 空头多 → bullish (+0.5 × 0.15 = +0.075)
├── Volatility: 正常 → ok
└── Market Bias: BEARISH → -0.1

综合得分: 0.30 + 0.04 + 0.09 + 0.075 - 0.1 = +0.405
强度: MODERATE (0.3 ≤ 0.405 < 0.6)
建议方向: LONG

T+0.1s: 状态转换 IDLE → ANALYZING
         信号入队 SignalQueue

T+0.2s: 状态转换 ANALYZING → WAITING_ENTRY
         收集市场快照

T+0.5s: Agent处理信号
├── 输入: signal + snapshot
├── MiniMax-M2.1 分析
│   ├── 分析多因子信号
│   ├── 检查当前仓位 (无)
│   ├── 查看最近操作记录
│   └── 综合判断
└── 输出: {"action": "open", "side": "long", "size": 100, "reason": "..."}

T+1s: 执行开仓
├── WEEX.place_order(side=long, size=100, type=market)
├── 订单成交
└── 状态转换 WAITING_ENTRY → IN_POSITION
    更新 PositionContext

T+60s: 持仓监控
├── 当前盈利: 8% of margin
├── DynamicTakeProfitRule: 未触发 (< 10%)
├── TrailingStopRule: 未激活 (< 15%)
└── ForceStopLossRule: 未触发 (> -25%)

T+120s: 利润达到12%
├── DynamicTakeProfitRule触发 (12% > 10%)
├── _last_signaled_threshold = 10
├── 状态转换 IN_POSITION → PROFIT_SIGNAL
├── Agent收到 PROFIT_SIGNAL_PROMPT
│   ├── 当前盈利: 12%
│   ├── 阈值级别: 10%
│   ├── 入场价格、持仓方向
│   └── 当前市场状态
├── Agent决定: stop_loss_price = 0.1234 (保护5%利润)
└── 执行止损单设置
    ├── 取消现有订单
    └── 新建 STOP_MARKET 订单

T+180s: 利润达到22%
├── DynamicTakeProfitRule再次触发 (22% > 20%)
├── _last_signaled_threshold = 20
├── Agent决定新止损价格: 0.1256 (保护15%利润)
└── 更新止损单

T+200s: TrailingStopRule激活
├── highest_pnl_percent = 25%
├── 激活阈值已达到 (25% > 15%)
└── 开始追踪

T+240s: 利润回撤到17%
├── TrailingStopRule计算:
│   ├── 最高利润: 25%
│   ├── 止损点: 25% - 7% = 18%
│   ├── 当前利润: 17% < 18%
│   └── 触发!
├── 立即强制平仓
├── 状态转换: IN_POSITION → RISK_OVERRIDE
├── 执行市价平仓
└── 状态转换: RISK_OVERRIDE → COOLDOWN
    记录交易结果 (win)

T+300s: 冷却期结束 (60秒)
└── 状态转换 COOLDOWN → IDLE
    等待下一个交易机会
```

---

## 八、数据流图

```
Binance WebSocket                    Binance REST API
      │                                     │
      │ 实时推送                            │ 每30秒轮询
      ▼                                     ▼
┌───────────┐                        ┌───────────┐
│ 1m Klines │                        │ Funding   │
│  Ticker   │                        │ L/S Ratio │
│ Orderbook │                        │ OI, Mark  │
└───────────┘                        └───────────┘
      │                                     │
      └──────────────┬──────────────────────┘
                     ▼
              ┌─────────────┐
              │  DataPool   │
              │ (Thread-Safe)│
              │   共享存储   │
              └─────────────┘
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ Indicators│  │Orchestrator│  │RiskMonitor│
│  (talipp) │  │ (5 Factors)│  │ (3 Rules) │
│ RSI,MACD  │  │   加权评分  │  │  风控检查  │
│ BB,ATR,MA │  └───────────┘  └───────────┘
└───────────┘        │              │
      │              ▼              ▼
      │       ┌─────────────┐ ┌───────────┐
      └──────▶│SignalQueue  │ │RiskAction │
              │ (去重队列)  │ │ (风控动作) │
              └─────────────┘ └───────────┘
                     │              │
                     └──────┬───────┘
                            ▼
                     ┌─────────────┐
                     │ MiniMax AI  │
                     │  Agent      │
                     │ (MiniMax-M2.1)│
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ WEEXExecutor│
                     │  交易执行   │
                     └─────────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
               ┌─────────┐   ┌─────────┐
               │ REST API│   │WebSocket│
               │ 下单/查询│   │ 仓位推送│
               └─────────┘   └─────────┘
```

---

## 九、关键配置参数

### 策略参数

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `MA_PERIOD` | 60 | orchestrator.py | 均线周期 (1H MA60) |
| `KLINE_INTERVAL` | 1h | orchestrator.py | 主要K线周期 |
| `RSI_OVERSOLD` | 30 | rsi_oversold.py | RSI超卖阈值 |
| `RSI_OVERBOUGHT` | 70 | rsi_oversold.py | RSI超买阈值 |
| `FUNDING_THRESHOLD` | 0.01% | funding_rate.py | 资金费率阈值 |
| `FUNDING_EXTREME` | 0.1% | funding_rate.py | 极端资金费率阈值 |
| `MARKET_BIAS` | BEARISH | orchestrator.py | 市场偏向 |

### 信号阈值

| 参数 | 值 | 说明 |
|------|------|------|
| `STRONG_SIGNAL_THRESHOLD` | 0.6 | 强信号阈值 |
| `MODERATE_SIGNAL_THRESHOLD` | 0.3 | 中等信号阈值 |
| `WEAK_SIGNAL_THRESHOLD` | 0.1 | 弱信号阈值 |

### 风控参数

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `FORCE_STOP_LOSS` | 25% | rules.py | 强制止损百分比 |
| `PROFIT_THRESHOLD` | 10% | rules.py | 盈利信号触发增量 |
| `TRAILING_ACTIVATION` | 15% | rules.py | 追踪止损激活点 |
| `TRAILING_DISTANCE` | 7% | rules.py | 追踪止损距离 |

### 时间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `COOLDOWN` | 60s | 平仓后冷却期 |
| `ANALYZING_TIMEOUT` | 30s | 分析超时 |
| `WAITING_ENTRY_TIMEOUT` | 60s | 等待开仓超时 |
| `WAITING_EXIT_TIMEOUT` | 60s | 等待平仓超时 |
| `STRATEGY_INTERVAL` | 5s | 策略评估间隔 |
| `RISK_INTERVAL` | 1s | 风控检查间隔 |
| `METRICS_INTERVAL` | 30s | 市场指标更新间隔 |

---

## 十、文件结构

```
src/ai_trading_team/
├── config.py                 # 配置管理
├── logging.py                # 日志设置
│
├── core/                     # 核心基础设施
│   ├── types.py              # 全局枚举 (Side, OrderType)
│   ├── events.py             # 事件系统
│   ├── data_pool.py          # 线程安全数据池
│   └── signal_queue.py       # 时间戳信号队列
│
├── data/                     # 数据模块 (Binance)
│   ├── models.py             # 数据模型
│   ├── binance/              # Binance REST + WebSocket
│   └── manager.py            # 数据管理器
│
├── indicators/               # 技术指标 (talipp)
│   ├── base.py               # 指标基类
│   ├── registry.py           # 指标注册表
│   └── technical.py          # RSI, MACD, BB, ATR
│
├── strategy/                 # 策略模块
│   ├── base.py               # 策略基类
│   ├── signals.py            # 信号定义
│   ├── orchestrator.py       # 多因子编排器
│   ├── state_machine.py      # 状态机
│   └── factors/              # 单因子策略
│       ├── ma_position.py    # MA60位置
│       ├── rsi_oversold.py   # RSI超买超卖
│       ├── funding_rate.py   # 资金费率
│       ├── long_short_ratio.py # 多空比
│       └── volatility.py     # 波动率
│
├── agent/                    # AI Agent模块
│   ├── llm.py                # LLM客户端
│   ├── prompts.py            # Prompt模板
│   ├── schemas.py            # 决策Schema
│   ├── commands.py           # 命令定义
│   └── trader.py             # 交易Agent
│
├── execution/                # 执行模块
│   ├── models.py             # Position, Order, Account
│   ├── base.py               # 交易所抽象接口
│   ├── dry_run.py            # 模拟执行器
│   ├── weex/                 # WEEX实现
│   └── manager.py            # 执行管理器
│
├── risk/                     # 风控模块
│   ├── rules.py              # 风控规则
│   ├── actions.py            # 风控动作
│   └── monitor.py            # 风控监控
│
└── audit/                    # 审计日志
    ├── models.py             # 日志模型
    ├── writer.py             # 本地写入器
    └── uploaders/            # 上传器
```

---

## 十一、核心设计原则

1. **多因子加权评分** - 5个因子综合计算，避免单一指标误判
2. **状态机管理** - 确保交易生命周期可预测，防止状态混乱
3. **双重风控** - 机械风控(强制止损/追踪止损) + AI决策(盈利止损)
4. **AI决策** - MiniMax-M2.1负责开仓判断和止损价格设置
5. **信号去重** - SignalQueue防止重复信号
6. **数据隔离** - Binance数据源 + WEEX执行，清晰分离
7. **操作记录** - 最近10次操作供Agent参考，避免重复错误
