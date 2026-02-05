- **POST** `/capi/v2/order/placeOrder`

Weight(IP): 2, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| client\_oid | String | Yes | Custom order ID (no more than 40 characters) |
| size | String | Yes | Order quantity (cannot be zero or negative). |
| type | String | Yes | 1: Open long, 2: Open short, 3: Close long, 4: Close short |
| order\_type | String | Yes | 0: Normal, 1: Post-Only, 2: Fill-Or-Kill, 3: Immediate Or Cancel |
| match\_price | String | Yes | 0: Limit price, 1: Market price |
| price | String | Yes | Order price (this is required for limit orders, and its accuracy and step size follow the futures information endpoint) |
| presetTakeProfitPrice | BigDecimal | No | Preset take-profit price |
| presetStopLossPrice | BigDecimal | No | Preset stop-loss price |
| marginMode | Integer | No | Margin mode   1: Cross Mode   3: Isolated Mode   Default is 1 (Cross Mode) |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/placeOrder" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol": "cmt_bchusdt","client_oid": "111111111222222","size": "1","type": "1","order_type": "0",
     "match_price": "0","price": "100","presetTakeProfitPrice": "105","presetStopLossPrice": "95"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| client\_oid | String | Client-generated order identifier |
| order\_id | String | Order ID |

**Response example**

```json
{
    "client_oid": null,
    "order_id": "596471064624628269"
}
```

- **POST** `/capi/v2/order/batchOrders`

Weight(IP): 5, Weight(UID): 10

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| marginMode | Integer | No | Margin mode   1: Cross Mode   3: Isolated Mode   Default is 1 (Cross Mode) |
| orderDataList | List | Yes | Maximum batch processing limit of 20 orders, with the same structure as the futures placing endpoint |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/placeOrder" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol": "cmt_bchusdt","orderDataList": [{
      "client_oid": "11111122222222","size": "1","type": "1","order_type": "0",
      "match_price": "0","price": "100"}]}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| order\_info | List | Order list |
| \>order\_id | String | Order ID |
| \>client\_oid | String | Customize order ID |
| \>result | Boolean | Order status |
| \>error\_code | String | Error code if order placement failed |
| \>error\_message | String | Error message if order placement failed |
| result | Boolean | Request result |

**Response example**

```json
{
    "order_info": [{
        "order_id": "596476148997685805",
        "client_oid": "order12346",
        "result": true,
        "error_code": "",
        "error_message": ""
    }],
    "result": true
}
```

- **POST** `/capi/v2/order/cancel_order`

Weight(IP): 2, Weight(UID): 3

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| orderId | String | No | Either Order ID or clientOid is required. |
| clientOid | String | No | Either Client customized ID or orderId is required. |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/cancel_order" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"orderId":"596471064624628269"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| order\_id | String | Order ID |
| client\_oid | String | Client identifier |
| result | Boolean | Cancellation status |
| err\_msg | String | Error message if cancellation failed |

**Response example**

```json
{
    "order_id": "596476148997685805",
    "client_oid": null,
    "result": true,
    "err_msg": null
}
```

- **POST** `/capi/v2/order/cancel_batch_orders`

Weight(IP): 5, Weight(UID): 10

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| ids | String\[\] | No | Either Order ID or cids is required. |
| cids | String\[\] | No | Either Client customized ID or ids is required. |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/cancel_order" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"ids": ["596471064624628269"]}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| result | Boolean | Processing result (success/failure) |
| orderIds | List | List of order IDs to be cancelled |
| clientOids | List | List of client order IDs |
| cancelOrderResultList | List | List of cancellation results |
| failInfos | List | List of failed cancellation info |

**CancelOrderResult Fields:**

| Parameter | Type | Description |
| --- | --- | --- |
| err\_msg | String | Error message if cancellation failed |
| order\_id | String | Order ID |
| client\_oid | String | Client order ID |
| result | boolean | Whether cancellation succeeded |

**Response example**

```json
{
  "result": true,
  "orderIds": ["596471064624628269"],
  "clientOids": [],
  "cancelOrderResultList": [
    {
      "err_msg": "",
      "order_id": "596471064624628269",
      "client_oid": "",
      "result": true
    }
  ],
  "failInfos": []
}
```

- **GET** `/capi/v2/order/detail`

Weight(IP): 2, Weight(UID): 2

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| orderId | String | Yes | Order ID |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/detail?orderId=596471064624628269" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| size | String | Order amount |
| client\_oid | String | Client identifier |
| createTime | long | Creation time   Unix millisecond timestamp |
| filled\_qty | String | Filled quantity |
| fee | String | Transaction fee |
| order\_id | String | Order ID |
| price | String | Order price |
| price\_avg | String | Average filled price |
| status | string | Order status   pending: The order has been submitted for matching, but the result has not been processed yet.   open: The order has been processed by the matching engine (order placed), and may have been partially filled.   filled: The order has been completely filled \[final state\].   canceling: The order is being canceled.   canceled: The order has been canceled. It may have been partially filled. \[final state\].   untriggered: The conditional order has not been triggered yet. |
| type | string | Order type   open\_long: Open long   open\_short: Open short   close\_long: Close long   close\_short: Close short   offset\_liquidate\_long: Reduce position, close long   offset\_liquidate\_short: Reduce position, close short   agreement\_close\_long: Agreement close long   agreement\_close\_short: Agreement close short   burst\_liquidate\_long: Liquidation close long   burst\_liquidate\_short: Liquidation close short |
| order\_type | string | Order type   normal: Regular limit order, valid until canceled.   postOnly: Maker-only order   fok: Fill or kill, must be completely filled or canceled immediately.   ioc: Immediate or cancel, fill as much as possible and cancel the remaining. |
| totalProfits | String | Total PnL |
| contracts | Integer | Order size in contract units |
| filledQtyContracts | Integer | Filled quantity in contract units |
| presetTakeProfitPrice | String | Preset take-profit price |
| presetStopLossPrice | String | Preset stop-loss price |

**Response example**

```json
{
  "symbol": "cmt_btcusdt",
  "size": "0.010000",
  "client_oid": "1763604184027_122",
  "createTime": "1763708511502",
  "filled_qty": "0.010000",
  "fee": "0.51357900",
  "order_id": "686643264626885530",
  "price": "0.0",
  "price_avg": "85596.5",
  "status": "filled",
  "type": "open_long",
  "order_type": "ioc",
  "totalProfits": "0",
  "contracts": 10000,
  "filledQtyContracts": 10000,
  "presetTakeProfitPrice": "100000.0",
  "presetStopLossPrice": "10000.0"
}
```

- **GET** `/capi/v2/order/history`

Weight(IP): 10, Weight(UID): 10

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair |
| pageSize | Integer | No | Items per page |
| createDate | Long | No | Creation time (must be ≤ 90 and cannot be negative)   Unix millisecond timestamp |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/history?symbol=cmt_bchusdt&pageSize=10&createDate=1742213506548" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Paramete | Type | Description |
| --- | --- | --- |
| Symbol | String | Trading pair |
| size | String | Order amount |
| client\_oid | String | Client identifier |
| createTime | String | Creation time   Unix millisecond timestamp |
| filled\_qty | String | Filled quantity |
| fee | String | Transaction fee |
| order\_id | String | Order ID |
| price | String | Order price |
| price\_avg | String | Average filled price |
| status | String | Order status   pending: The order has been submitted for matching, but the result has not been processed yet.   open: The order has been processed by the matching engine (order placed), and may have been partially filled.   filled: The order has been completely filled \[final state\].   canceling: The order is being canceled.   canceled: The order has been canceled. It may have been partially filled. \[final state\].   untriggered: The conditional order has not been triggered yet. |
| type | String | Order type   open\_long: Open long   open\_short: Open short   close\_long: Close long   close\_short: Close short   offset\_liquidate\_long: Reduce position, close long   offset\_liquidate\_short: Reduce position, close short   agreement\_close\_long: Agreement close long   agreement\_close\_short: Agreement close short   burst\_liquidate\_long: Liquidation close long   burst\_liquidate\_short: Liquidation close short |
| order\_type | String | Order type   normal: Regular limit order, valid until canceled.   postOnly: Maker-only order   fok: Fill or kill, must be completely filled or canceled immediately.   ioc: Immediate or cancel, fill as much as possible and cancel the remaining. |
| totalProfits | String | Total PnL |
| contracts | Integer | Order size in contract units |
| filledQtyContracts | Integer | Filled quantity in contract units |
| presetTakeProfitPrice | String | Preset take-profit price |
| presetStopLossPrice | String | Preset stop-loss price |

**Response example**

```json
[
  {
    "symbol": "cmt_btcusdt",
    "size": "0.010000",
    "client_oid": "1478527825782587",
    "createTime": "1764505776340",
    "filled_qty": "0.010000",
    "fee": "0.54731220",
    "order_id": "689987235725968026",
    "price": "0.0",
    "price_avg": "91218.7",
    "status": "filled",
    "type": "open_long",
    "order_type": "ioc",
    "totalProfits": "0",
    "contracts": 10000,
    "filledQtyContracts": 10000,
    "presetTakeProfitPrice": null,
    "presetStopLossPrice": null
  }
]
```

- **GET** `/capi/v2/order/current`

Weight(IP): 2, Weight(UID): 2

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair |
| orderId | Long | No | OrderId |
| startTime | Long | No | The record start time for the query   Unix millisecond timestamp |
| endTime | Long | No | The end time of the record for the query   Unix millisecond timestamp |
| limit | Integer | No | Limit number default 100 max 100 |
| page | Integer | No | Page number default 0 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/current?symbol=cmt_bchusdt" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| size | String | Order amount |
| client\_oid | String | Client identifier |
| createTime | String | Creation time   Unix millisecond timestamp |
| filled\_qty | String | Filled quantity |
| fee | String | Transaction fee |
| order\_id | String | Order ID |
| price | String | Order price |
| price\_avg | String | Average filled price |
| status | String | Order status   pending: The order has been submitted for matching, but the result has not been processed yet.   open: The order has been processed by the matching engine (order placed), and may have been partially filled.   filled: The order has been completely filled \[final state\].   canceling: The order is being canceled.   canceled: The order has been canceled. It may have been partially filled. \[final state\].   untriggered: The conditional order has not been triggered yet. |
| type | String | Order type   open\_long: Open long   open\_short: Open short   close\_long: Close long   close\_short: Close short   offset\_liquidate\_long: Reduce position, close long   offset\_liquidate\_short: Reduce position, close short   agreement\_close\_long: Agreement close long   agreement\_close\_short: Agreement close short   burst\_liquidate\_long: Liquidation close long   burst\_liquidate\_short: Liquidation close short |
| order\_type | String | Order type   normal: Regular limit order, valid until canceled.   postOnly: Maker-only order   fok: Fill or kill, must be completely filled or canceled immediately.   ioc: Immediate or cancel, fill as much as possible and cancel the remaining. |
| totalProfits | String | Total PnL |
| contracts | Integer | Order size in contract units |
| filledQtyContracts | Integer | Filled quantity in contract units |
| presetTakeProfitPrice | String | Preset take-profit price |
| presetStopLossPrice | String | Preset stop-loss price |

**Response example**

```json
[
  {
    "symbol": "cmt_btcusdt",
    "size": "0.010000",
    "client_oid": "175287228528278",
    "createTime": "1764505828770",
    "filled_qty": "0",
    "fee": "0",
    "order_id": "689987455633326746",
    "price": "88888.0",
    "price_avg": "0",
    "status": "open",
    "type": "open_long",
    "order_type": "normal",
    "totalProfits": "0",
    "contracts": 10000,
    "filledQtyContracts": 0,
    "presetTakeProfitPrice": null,
    "presetStopLossPrice": null
  }
]
```

- **GET** `/capi/v2/order/fills`

Weight(IP): 5, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair name |
| orderId | Long | No | Order ID |
| startTime | Long | No | Start timestamp   Unix millisecond timestamp |
| endTime | Long | No | End timestamp   Unix millisecond timestamp |
| limit | Long | No | Number of queries: Maximum: 100, default: 100 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/fills?symbol=cmt_bchusdt&orderId=596471064624628269" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| list | List | Transaction details |
| \> tradeId | Integer | Filled order ID |
| \> orderId | Integer | Associated order ID |
| \> symbol | String | Trading pair name |
| \> marginMode | String | Margin mode |
| \> separatedMode | String | Separated mode |
| \> positionSide | String | Position direction |
| \> orderSide | String | Order direction |
| \> fillSize | String | Actual filled quantity |
| \> fillValue | String | Actual filled value |
| \> fillFee | String | Actual trading fee |
| \> liquidateFee | String | Closing fee |
| \> realizePnl | String | Actual realized PnL |
| \> direction | String | Actual execution direction |
| \> liquidateType | String | Liquidation order type |
| \> legacyOrdeDirection | String | Compatible with legacy order direction types |
| \> createdTime | Integer | Timestamp   Unix millisecond timestamp |
| nextFlag | Boolean | Whether more pages exist |
| totals | Integer | Total entries |

**Response example**

- **POST** `/capi/v2/order/plan_order`

Weight(IP): 2, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| client\_oid | String | Yes | Custom order ID (≤40 chars, no special characters), must be unique in maker ordersIf left empty, the system will automatically assign a value. |
| size | String | Yes | Order quantity in lots (cannot be zero or negative) |
| type | String | Yes | 1: Open long 2. Open short 3. Close long 4. Close short |
| match\_type | String | Yes | 0: Limit price, 1: Market price |
| execute\_price | String | Yes | Execution price |
| trigger\_price | String | Yes | Trigger price |
| marginMode | Integer | No | Margin mode   1: Cross Mode   3: Isolated Mode   Default is 1 (Cross Mode) |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/placeOrder" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol": "cmt_bchusdt","client_oid": "11111111111111","size": "1",
   "type": "1","match_type": "1","execute_price": "100","trigger_price": "100"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| client\_oid |  | Client identifier |
| order\_id |  | Conditional order ID |

**Response example**

```json
{
    "client_oid": null,
    "order_id": "596480271352594989"
}
```

- **POST** `/capi/v2/order/cancel_plan`

Weight(IP): 2, Weight(UID): 3

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| orderId | String | Yes | Order ID |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/cancel_plan" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"orderId":"596471064624628269"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| order\_id | String | Order ID |
| client\_oid | String | Client identifier |
| result | Boolean | Whether the cancellation was successful |
| err\_msg | String | Error message if the cancellation failed |

**Response example**

```json
{
    "order_id": "596480271352594989",
    "client_oid": null,
    "result": true,
    "err_msg": null
}
```

- **GET** `/capi/v2/order/currentPlan`

Weight(IP): 3, Weight(UID): 3

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair |
| orderId | Long | No | OrderId (The filter will be bypassed if invalid characters are input.) |
| startTime | Long | No | The record start time for the query   Unix millisecond timestamp |
| endTime | Long | No | The end time of the record for the query   Unix millisecond timestamp |
| limit | Integer | No | Limit number default 100 max 100 |
| page | Integer | No | Page number default 0 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/currentPlan?symbol=cmt_bchusdt" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| size | String | Order amount |
| client\_oid | String | Client identifier |
| createTime | String | Creation time   Unix millisecond timestamp |
| filled\_qty | String | Filled quantity |
| fee | String | Transaction fee |
| order\_id | String | Order ID |
| price | String | Order price |
| price\_avg | String | Average filled price |
| status | String | Order status: -1: Canceled. 0: Pending. 1: Partially filled. 2: Filled |
| type | String | Order Type: 1. Open long. 2: Open short. 3: Close long. 4: Close short. 5: Partial close long. 6: Partial close short. 7: Auto-deleveraging (close long). 8: Auto-deleveraging (close short). 9: Liquidation (close long). 10. Liquidation (close short). |
| order\_type | String | Order type: 0: Normal order. 1: Post-only. 2: Fill-Or-Kill (FOK) order. 3: Immediate-Or-Cancel (IOC) order. |
| totalProfits | String | Total PnL |
| triggerPrice | String | Trigger price |
| triggerPriceType | String | Trigger price type |
| triggerTime | String | Trigger time   Unix millisecond timestamp |
| presetTakeProfitPrice | String | Preset take-profit price |
| presetStopLossPrice | String | Preset stop-loss price |

**Response example**

```json
[{
  "symbol": "cmt_btcusdt",
  "size": "1",
  "client_oid": "1234567890",
  "createTime": "1742213506548",
  "filled_qty": "0.5",
  "fee": "0.01",
  "order_id": "461234125",
  "price": "50000.00",
  "price_avg": "49900.00",
  "status": "1",
  "type": "1",
  "order_type": "0",
  "totalProfits": "200.00",
  "triggerPrice": "48000.00",
  "triggerPriceType": "LIMIT",
  "triggerTime": "1742213506548",
  "presetTakeProfitPrice": null,
  "presetStopLossPrice": null
}]
```

- **GET** `/capi/v2/order/historyPlan`

Weight(IP): 5, Weight(UID): 10

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| startTime | Long | No | Start time   Unix millisecond timestamp |
| endTime | Long | No | End time   Unix millisecond timestamp |
| delegateType | Integer | No | Order type: 1: Open long. 2: Open short. 3: Close long. 4: Close short. |
| pageSize | Integer | No | Items per page, ranging from 1 to 100, default is 100. |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/order/historyPlan?symbol=cmt_bchusdt&delegateType=2&startTime=1742213127794&endTime=1742213506548" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| size | String | Order quantity |
| client\_oid | String | Client identifier |
| createTime | String | Creation time   Unix millisecond timestamp |
| filled\_qty | String | Filled quantity |
| fee | String | Transaction fee |
| order\_id | String | Order ID |
| price | String | Order price |
| price\_avg | String | Average filled price |
| status | String | Order status: -1: Canceled. 0: Pending. 1: Partially filled. 2: Filled |
| type | String | Order Type: 1. Open long. 2: Open short. 3: Close long. 4: Close short. 5: Partial close long. 6: Partial close short. 7: Auto-deleveraging (close long). 8: Auto-deleveraging (close short). 9: Liquidation (close long). 10. Liquidation (close short). |
| order\_type | String | Order type: 0: Normal order. 1: Post-only. 2: Fill-Or-Kill (FOK) order. 3: Immediate-Or-Cancel (IOC) order. |
| totalProfits | String | Total PnL |
| triggerPrice | String | Trigger price |
| triggerPriceType | String | Trigger price type |
| triggerTime | String | Trigger time   Unix millisecond timestamp |
| presetTakeProfitPrice | String | Preset take-profit price |
| presetStopLossPrice | String | Preset stop-loss price |

**Response example**

```json
{
    "list": [{
        "symbol": "cmt_btcusdt",
        "size": "1",
        "client_oid": "1234567890",
        "createTime": "1742213506548",
        "filled_qty": "0.5",
        "fee": "0.01",
        "order_id": "461234125",
        "price": "50000.00",
        "price_avg": "49900.00",
        "status": "1",
        "type": "1",
        "order_type": "0",
        "totalProfits": "200.00",
        "triggerPrice": "48000.00",
        "triggerPriceType": "",
        "triggerTime": "1742213506548",
        "presetTakeProfitPrice": null,
        "presetStopLossPrice": null
    }],
    "nextPage": false
}
```

- **POST** `/capi/v2/order/closePositions`

Weight(IP): 40, Weight(UID): 50

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair. If not provided, all positions will be closed at market price |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/closePositions" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol": "cmt_btcusdt"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| positionId | Long | Position ID |
| success | Boolean | Whether the position was successfully closed |
| successOrderId | Long | Order ID if successful (0 if failed) |
| errorMessage | String | Error message if the close position failed |

**Response example**

```json
[
  {
    "positionId": 690800371848708186,
    "successOrderId": 696023766399976282,
    "errorMessage": "",
    "success": true
  }
]
```

- **POST** `/capi/v2/order/cancelAllOrders`

Weight(IP): 40, Weight(UID): 50

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair. If not provided, orders for all trading pairs will be cancelled |
| cancelOrderType | String | Yes | Order type to cancel:   `normal`: Cancel normal orders   `plan`: Cancel trigger/plan orders |

**Request example**

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| orderId | Long | Order ID |
| success | Boolean | Whether the order was cancelled successfully |

**Response example**

```json
[
  {
    "orderId": 696026685023191898,
    "success": true
  }
]
```

- **POST** `/capi/v2/order/placeTpSlOrder`

Weight(IP): 2, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| clientOrderId | String | Yes | Custom order ID (no more than 40 characters) |
| planType | String | Yes | TP/SL type:   `profit_plan`: Take-profit plan order   `loss_plan`: Stop-loss plan order |
| triggerPrice | String | Yes | Trigger price |
| executePrice | String | No | Execution price. If not provided or set to 0, market price will be used. Value > 0 means limit price |
| size | String | Yes | Order quantity |
| positionSide | String | Yes | Position direction:   `long`: Long position   `short`: Short position |
| marginMode | Integer | No | Margin mode:   1: Cross Mode   3: Isolated Mode   Default is 1 (Cross Mode) |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/placeTpSlOrder" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol": "cmt_btcusdt", "clientOrderId": "123456789", "planType": "profit_plan",
        "triggerPrice": "50000", "executePrice": "0", "size": "1", "positionSide": "long", "marginMode": 1}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| success | Boolean | Whether the TP/SL order was placed successfully |
| orderId | Long | Order ID (0 if failed) |

**Response example**

```json
[
  {
    "orderId": 696073048050107226,
    "success": true
  }
]
```

- **POST** `/capi/v2/order/modifyTpSlOrder`

Weight(IP): 2, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| orderId | Long | Yes | Order ID of the TP/SL order to modify |
| triggerPrice | String | Yes | New trigger price |
| executePrice | String | No | New execution price. If not provided or set to 0, market price will be used. Value > 0 means limit price |
| triggerPriceType | Integer | No | Trigger price type:   1: Last price   3: Mark price   Default is 1 (Last price) |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/modifyTpSlOrder" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"orderId": 596471064624628269, "triggerPrice": "51000", "executePrice": "0", "triggerPriceType": 1}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| code | String | Response code, "00000" indicates success |
| msg | String | Response message |
| requestTime | Long | return time   Unix millisecond timestamp |
| data | Object | Response data (null for this endpoint) |

**Response example**

```json
{
  "code": "00000",
  "msg": "success",
  "requestTime": 1765956639711,
  "data": ""
}
```