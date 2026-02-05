## 数据获取

从binance获取最新和实时数据

K线数据

启动时获取 5m 15m 1h 4h 最近的500条历史K线数据，后续数据使用 ws 实时数据补齐

Ticker 数据

订阅 ws 实时推送的ticker数据

Orderbook 数据

订阅 ws 实时推送的orderbook数据

Trades 数据

订阅 ws 实时推送的trades数据

LongShortRatio 数据

订阅 ws 实时推送的longshortratio数据，如果没有就每隔1分钟轮询一次binance rest api 获取一次

FundingRate 数据

你可以订阅实时的 markprice 和 funding rate 数据，如果没有就每隔1分钟轮询一次binance rest api 获取一次

OpenInterest 数据

你可以订阅实时的 openinterest 数据，如果没有就每隔1分钟轮询一次binance rest api 获取一次

Liquidation 数据

订阅 ws 实时推送的liquidation数据

从 weex 获取用户实时的账户、订单、仓位数据，除了 openorders 以外也要获取最近的 10 次操作记录

模拟状态下全都本地模拟操作记录，以及账户、挂单、仓位信息，无需访问 weex

## 指标模块

使用 talipp 计算技术指标，每次有相关依赖数据的时候都要重新计算指标

talipp 是支持增量计算的，所以消耗很小

计算以下指标：
SMA(60)
RSI(14)
ATR(14)
Bollinger Bands(20)
MACD(12, 26, 9)
ADX(14)

均包含 5m 15m 1h 4h 四个时间周期的指标值

## 信号模块

发生以下事件时固定发送信号给 agent：

1. RSI(14) 进入超卖区
2. RSI(14) 进入超买区
3. MACD(12, 26, 9) 金叉
4. MACD(12, 26, 9) 死叉
6. Bollinger Bands(20) 突破
7. close 上穿或者下穿 SMA(60) 时

上述信号均包含 5m 15m 1h 4h 四个时间周期的信号

除此之外还有

OI 在5分钟内变化超过5%时
多空比在5分钟内变化超过5%时
发生1M美金价值以上的liquidation时

此外

当用户的盈亏每增加或减少 5% 时，注意这里的盈亏百分比，应该用盈亏的美金价值除以用户当前的保证金来计算
当风险模块进行强制止损时，也要发送信号给 agent
openorders 状态发生变化时也要发送信号给 agent

在实际运行过程中，信号可能会出现重叠，因此我们需要维护一个信号队列，在agent对于上一个信号做出判断之前，如果有新的信号产生，则将新的信号加入队列，下一个信号发送给agent时，需要在context中添加它对于之前信号做出的判断结果，以便于agent进行综合判断，不至于错乱。

每次发送信号给 agent 的时候，必须使用最新数据的 snapshot 添加到 context 中，作为 agent 的输入。