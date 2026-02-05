## Overview

WebSocket is a new protocol in HTML5 that enables full-duplex communication between clients and servers, allowing rapid bidirectional data transmission. Through a simple handshake, a connection can be established between client and server, enabling the server to actively push information to the client based on business rules. Its advantages include:

- Small header size (~2 bytes) during data transmission between client and server
- Both client and server can actively send data
- Eliminates the need for repeated TCP connection setup/teardown, conserving bandwidth and server resources
- Strongly recommended for developers to obtain market data, order book depth, and other information

| Domain | WebSocket API | Recommended Use |
| --- | --- | --- |
| Public Channel | wss://ws-contract.weex.com/v2/ws/public | Primary domain, public channels |
| Private Channel | wss://ws-contract.weex.com/v2/ws/private | Primary domain, private channels |

## Connection

Connection Specifications:

- Connection limit: 300 connection requests/IP/5 minutes, maximum 100 concurrent connections per IP
- Subscription limit: 240 operations/hour/connection, maximum 100 channels per connection
- Public channel requirement: Public channel connections require header authentication(User-Agent)
- Private channel requirement: Private channel connections require header authentication
- To maintain stable and effective connections, we recommend:
- After successful WebSocket connection establishment, the server will periodically send Ping messages to the client in the format: `{"event":"ping","time":"1693208170000"}`, where "time" represents the server's timestamp. Upon receiving this message, the client should respond with a Pong message: `{"event":"pong","time":"1693208170000"}`. The server will actively terminate connections that fail to respond more than 5 times.

## Header Authentication for Private Channels

**User-Agent**:Client identification

**ACCESS-KEY**: Unique identifier for API user authentication (requires application)

**ACCESS-PASSPHRASE**: Password for the API Key

**ACCESS-TIMESTAMP**: Unix Epoch timestamp in milliseconds (expires after 30 seconds, must match signature timestamp)

**ACCESS-SIGN**: Signature string generated as follows:

The message (string to be signed) consists of: timestamp + requestPath

Example timestamp (in milliseconds):`const timestamp = '' + Date.now()`

Where requestPath is `/v2/ws/private`

**Signature Generation Process**

1. Encrypt the message string using HMAC SHA256 with the secret key:
	- Signature = hmac\_sha256(secretkey, Message)
2. Encode the Signature using Base64:
	- Signature = base64.encode(Signature)

## Subscription

Subscription Specification:

## Unsubscription

Unsubscription Specification:

**Description**  
Retrieves real-time market data including latest price, best bid/ask prices, and 24-hour trading volume. Updates occur within 100-300ms when changes occur (trades, bid/ask updates).

**Push Data Parameters**

| Field | Type | Description |
| --- | --- | --- |
| event | String | Push action |
| channel | String | Channel name (e.g. ticker.cmt\_btcusdt) |
| data | List | Market data array |
| \>symbol | String | Product ID |
| \>priceChange | String | Price change amount |
| \>priceChangePercent | String | Price change percentage |
| \>trades | String | 24h trade count |
| \>size | String | 24h trading volume |
| \>value | String | 24h trading value |
| \>high | String | 24h highest price |
| \>low | String | 24h lowest price |
| \>lastPrice | String | Latest traded price |
| \>markPrice | String | Current mark price |

**Push Data Example**

```json
{  
  "event": "payload",  
  "channel": "ticker.cmt_btcusdt",  
  "data": [  
    {  
      "symbol": "cmt_btcusdt",  
      "priceChange": "-2055.6",  
      "priceChangePercent": "-0.019637",  
      "trades": "28941",  
      "size": "176145.66489",  
      "value": "18115688543.1",  
      "high": "104692.2",  
      "low": "100709.6",  
      "lastPrice": "102623.9",  
      "markPrice": "102623.9"  
    }  
  ]  
}
```

**Description**  
Order K-line channel

**Push Data Example**

```json
{
  "event": "payload",
  "type": "change",
  "channel": "kline.LAST_PRICE.cmt_btcusdt.MINUTE_1",
  "data": [
    {
      "symbol": "cmt_btcusdt",
      "klineTime": "1747125660000",
      "size": "23.76600",
      "value": "2442678.713400",
      "high": "102784.6",
      "low": "102760.6",
      "open": "102760.6",
      "close": "102764.0"
    }
  ]
}
```

**Description**

Retrieves order book depth data

Upon successful subscription, a full snapshot will be pushed initially (depthType=SNAPSHOT), followed by incremental updates (depthType=CHANGED).

**Push Data Example**

```json
{  
  "event": "payload",  
  "channel": "depth.cmt_btcusdt.15",  
  "data": [  
    {  
      "startVersion": "3644174246",  
      "endVersion": "3644174270",  
      "level": 15,  
      "depthType": "CHANGED",  
      "symbol": "cmt_btcusdt",  
      "asks": [  
        {  
          "price": "103436.1",  
          "size": "0.91500"  
        },  
        {  
          "price": "103436.3",  
          "size": "1.95800"  
        },  
        {  
          "price": "103436.5",  
          "size": "0"  
        },  
        {  
          "price": "103436.6",  
          "size": "1.08300"  
        },  
        {  
          "price": "103436.7",  
          "size": "7.64700"  
        },  
        {  
          "price": "103436.9",  
          "size": "7.23100"  
        },  
        {  
          "price": "103437.0",  
          "size": "0"  
        },  
        {  
          "price": "103437.2",  
          "size": "0"  
        }  
      ],  
      "bids": [  
        {  
          "price": "103435.9",  
          "size": "2.40500"  
        },  
        {  
          "price": "103435.7",  
          "size": "0"  
        },  
        {  
          "price": "103435.6",  
          "size": "0.32700"  
        },  
        {  
          "price": "103435.5",  
          "size": "0"  
        },  
        {  
          "price": "103435.2",  
          "size": "3.19400"  
        },  
        {  
          "price": "103434.8",  
          "size": "10.25000"  
        },  
        {  
          "price": "103434.5",  
          "size": "11.13900"  
        }  
      ]  
    }  
  ]  
}
```

**Description**

Platform trade channel (taker orders)

**Push Response Example**

```json
{
  "event": "payload",
  "channel": "trades.cmt_btcusdt",
  "data": [
    {
      "time": "1747131727502",
      "price": "103337.5",
      "size": "0.01600",
      "value": "1653.400000",
      "buyerMaker": false
    }
  ]
}
```

**Description**

Subscribe to the account channel

**Push Data Parameters**

| Field | Type | Description |
| --- | --- | --- |
| coinId | String | Currency ID |
| marginMode | String | Margin mode |
| crossContractId | String | When marginMode=CROSS, represents the associated contract ID in cross margin mode. Otherwise, it is 0. |
| isolatedPositionId | String | When marginMode=ISOLATED, represents the associated position ID in isolated margin mode. Otherwise, it is 0. |
| amount | String | Collateral amount |
| pendingDepositAmount | String | Pending deposit amount |
| pendingWithdrawAmount | String | Pending withdrawal amount |
| pendingTransferInAmount | String | Pending transfer-in amount |
| pendingTransferOutAmount | String | Pending transfer-out amount |
| isLiquidating | String | Whether liquidation is triggered (under forced liquidation) |
| legacyAmount | String | Balance field (display only, not used in calculations) |
| cumDepositAmount | String | Cumulative deposit amount |
| cumWithdrawAmount | String | Cumulative withdrawal amount |
| cumTransferInAmount | String | Cumulative transfer-in amount |
| cumTransferOutAmount | String | Cumulative transfer-out amount |
| cumMarginMoveInAmount | String | Cumulative margin transfer-in amount |
| cumMarginMoveOutAmount | String | Cumulative margin transfer-out amount |
| cumPositionOpenLongAmount | String | Cumulative collateral amount for opening long positions |
| cumPositionOpenShortAmount | String | Cumulative collateral amount for opening short positions |
| cumPositionCloseLongAmount | String | Cumulative collateral amount for closing long positions |
| cumPositionCloseShortAmount | String | Cumulative collateral amount for closing short positions |
| cumPositionFillFeeAmount | String | Cumulative trade fee amount |
| cumPositionLiquidateFeeAmount | String | Cumulative liquidation fee amount |
| cumPositionFundingAmount | String | Cumulative funding fee amount |
| cumOrderFillFeeIncomeAmount | String | Cumulative order fee income amount |
| cumOrderLiquidateFeeIncomeAmount | String | Cumulative liquidation fee income amount |
| createdTime | String | Creation time |
| updatedTime | String | Update time |

**Push Response Example**

**Description**

Subscribe to position channel

**Push Data Parameters**

| Field | Type | Description |
| --- | --- | --- |
| id | String | Position ID |
| coinId | String | Collateral currency ID |
| contractId | String | Contract ID |
| side | String | Position direction (LONG/SHORT) |
| marginMode | String | Margin mode for current position |
| separatedMode | String | Position separation mode |
| separatedOpenOrderId | String | Separated position opening order ID |
| leverage | String | Position leverage |
| size | String | Position size |
| openValue | String | Opening value |
| openFee | String | Opening fee |
| fundingFee | String | Funding fee |
| isolatedMargin | String | Isolated margin |
| isAutoAppendIsolatedMargin | String | Whether auto-append isolated margin |
| cumOpenSize | String | Cumulative opening size |
| cumOpenValue | String | Cumulative opening value |
| cumOpenFee | String | Cumulative opening fees |
| cumCloseSize | String | Cumulative closing size |
| cumCloseValue | String | Cumulative closing value |
| cumCloseFee | String | Cumulative closing fees |
| cumFundingFee | String | Cumulative settled funding fees |
| cumLiquidateFee | String | Cumulative liquidation fees |
| createdMatchSequenceId | String | Matching engine sequence ID at creation |
| updatedMatchSequenceId | String | Matching engine sequence ID at update |
| createdTime | String | Creation timestamp |
| updatedTime | String | Update timestamp |

**Push Response Example**

**Description**

Subscribe to trade details information

**Push Data Parameters**

| Field | Type | Description |
| --- | --- | --- |
| id | String | Unique identifier |
| coinId | String | Collateral currency ID |
| contractId | String | Contract ID |
| orderId | String | Order ID |
| marginMode | String | Margin mode |
| separatedMode | String | Position separation mode |
| separatedOpenOrderId | String | Separated position creation order ID (exists only when separated\_mode=SEPARATED) |
| positionSide | String | Position direction (always UNKNOWN for one-way positions) |
| orderSide | String | Buy/Sell direction |
| fillSize | String | Actual filled quantity |
| fillValue | String | Actual filled value |
| fillFee | String | Actual transaction fee (precise value) |
| liquidateFee | String | Liquidation fee (if trade is a liquidation) |
| realizePnl | String | Realized profit/loss (only appears for closing trades) |
| direction | String | Execution direction (MAKER/TAKER) |
| createdTime | String | Creation timestamp |
| updatedTime | String | Update timestamp |

**Push Response Example**

**Description**

Subscribe to order channel

**Push Data Parameters**

| Field | Type | Description |
| --- | --- | --- |
| id | String | Order ID (value > 0) |
| coinId | String | Collateral currency ID |
| contractId | String | Contract ID |
| marginMode | String | Margin mode |
| separatedMode | String | Position separation mode |
| separatedOpenOrderId | String | Separated position creation order ID (exists only when separated\_mode=SEPARATED) |
| positionSide | String | Position direction (always UNKNOWN for one-way positions) |
| orderSide | String | Buy/Sell direction |
| price | String | Order price (worst acceptable price) |
| size | String | Order quantity |
| clientOrderId | String | Client custom ID for idempotency check |
| type | String | Order type |
| timeInForce | String | Order execution strategy (meaningful when type is LIMIT/STOP\_LIMIT/TAKE\_PROFIT\_LIMIT) |
| reduceOnly | String | Whether reduce-only order |
| triggerPrice | String | Trigger price (meaningful for STOP\_LIMIT/STOP\_MARKET/TAKE\_PROFIT\_LIMIT/TAKE\_PROFIT\_MARKET orders, 0 means empty) |
| triggerPriceType | String | Price type: last price \[default\], mark price (meaningful for STOP\_LIMIT/STOP\_MARKET/TAKE\_PROFIT\_LIMIT/TAKE\_PROFIT\_MARKET orders) |
| isPositionTpsl | String | Whether position take-profit/stop-loss order |
| orderSource | String | Order source |
| openTpslParentOrderId | String | Opening order ID for position take-profit/stop-loss orders |
| isSetOpenTp | String | Whether set open take-profit |
| openTpParam | String | Open take-profit parameters |
| isSetOpenSl | String | Whether set open stop-loss |
| openSlParam | String | Open stop-loss parameters |
| leverage | String | Leverage multiplier when opening position |
| takerFeeRate | String | Taker fee rate when placing order |
| makerFeeRate | String | Maker fee rate when placing order |
| feeDiscount | String | Fee discount rate when placing order |
| liquidateFeeRate | String | Liquidation fee rate when placing order |
| status | String | Order status |
| triggerTime | String | Conditional order trigger time |
| triggerPriceTime | String | Conditional order trigger price time |
| triggerPriceValue | String | Conditional order trigger price value |
| cancelReason | String | Order cancellation reason |
| latestFillPrice | String | Latest filled price of current order |
| maxFillPrice | String | Highest filled price of current order |
| minFillPrice | String | Lowest filled price of current order |
| cumFillSize | String | Cumulative filled quantity after matching |
| cumFillValue | String | Cumulative filled value after matching |
| cumFillFee | String | Cumulative transaction fee after matching |
| cumLiquidateFee | String | Cumulative liquidation fee |
| cumRealizePnl | String | Cumulative realized profit/loss |
| createdTime | String | Creation time |
| updatedTime | String | Update time |

**Push Response Example**

```json
{  
  "type": "trade-event",  
  "channel": "orders",  
  "event": "payload",  
  "msg": {  
    "msgEvent": "OrderUpdate",  
    "version": 46654,  
    "data": {  
      "order": [  
        {  
          "id": "617414920861909658",  
          "coinId": "USDT",  
          "contractId": "cmt_btcusdt",  
          "marginMode": "SHARED",  
          "separatedMode": "COMBINED",  
          "separatedOpenOrderId": "0",  
          "positionSide": "LONG",  
          "orderSide": "BUY",  
          "price": "0.0",  
          "size": "0.10000",  
          "clientOrderId": "1747203186927fpiZrpAEkOlH3ygdwfJpowP0HeXVer7JFxxmIohyCMPXqKCz74s",  
          "type": "MARKET",  
          "timeInForce": "IMMEDIATE_OR_CANCEL",  
          "reduceOnly": false,  
          "triggerPrice": "0",  
          "triggerPriceType": "UNKNOWN_PRICE_TYPE",  
          "orderSource": "WEB",  
          "openTpslParentOrderId": "0",  
          "leverage": "20",  
          "takerFeeRate": "0.0006",  
          "makerFeeRate": "0.0002",  
          "feeDiscount": "1",  
          "liquidateFeeRate": "0.01",  
          "status": "PENDING",  
          "triggerTime": "0",  
          "triggerPriceTime": "0",  
          "triggerPriceValue": "0",  
          "cancelReason": "UNKNOWN_ORDER_CANCEL_REASON",  
          "latestFillPrice": "0",  
          "maxFillPrice": "0",  
          "minFillPrice": "0",  
          "cumFillSize": "0",  
          "cumFillValue": "0",  
          "cumFillFee": "0",  
          "cumLiquidateFee": "0",  
          "cumRealizePnl": "0",  
          "createdTime": "1747203188148",  
          "updatedTime": "1747203188148",  
          "positionTpsl": false,  
          "setOpenTp": false,  
          "setOpenSl": false  
        }  
      ]  
    },  
    "time": 1747203188148  
  }  
}
```

All APIs may return exceptions.

The following are possible error codes returned by the API

| Error Message | Error Code | HTTP Status Code |
| --- | --- | --- |
| Header "ACCESS\_KEY" is required | 40001 | 400 |
| Header "ACCESS\_SIGN" is required | 40002 | 400 |
| Header "ACCESS\_TIMESTAMP" is required | 40003 | 400 |
| Invalid ACCESS\_TIMESTAMP | 40005 | 400 |
| Invalid ACCESS\_KEY | 40006 | 400 |
| Invalid Content\_Type, use "application/json" format | 40007 | 400 |
| Request timestamp expired | 40008 | 400 |
| API verification failed | 40009 | 400 |
| Too many requests | 429 | 429 |
| Header "ACCESS\_PASSPHRASE" is required | 40011 | 400 |
| Incorrect API key/Passphrase | 40012 | 400 |
| Account frozen | 40013 | 400 |
| Invalid permissions | 40014 | 400 |
| System error | 40015 | 400 |
| Parameter validation failed | 40017 | 400 |
| Invalid IP request | 40018 | 400 |
| Parameter cannot be empty | 40019 | 400 |
| Parameter is invalid | 40020 | 400 |
| API permission disabled | 40753 | 400 |
| Insufficient permissions for this operation | 40022 | 403 |
| Not have permission to trade this pair. | 50003 | 400 |
| Not have permission to access this API. | 50004 | 400 |
| Order does not exist | 50005 | 400 |
| Leverage cannot exceed the limit | 50007 | 400 |

## Error Response Format

All errors return a JSON response in the following format:

```json
{
  "code": "error code",
  "msg": "error description",
  "requestTime": 1765776384556,
  "data": null
}
```

### Example 1: Parameter Error

```json
{
  "code": "40020",
  "msg": "Parameter symbol is invalid",
  "requestTime": 1765776928145,
  "data": null
}
```