- **GET** `/capi/v2/market/time`

Weight(IP): 1

**Request parameters**

NONE

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/time"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| epoch | String | Unix timestamp in UTC time zone, represented as a decimal number of seconds |
| iso | String | ISO 8601 standard time format |
| timestamp | Long | Server time   Unix millisecond timestamp |

**Response example**

```json
{
    "epoch": "1716710918.113",
    "iso": "2024-05-26T08:08:38.113Z",
    "timestamp": 1716710918113
}
```

- **GET** `/capi/v2/market/contracts`

Weight(IP): 10

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/contracts?symbol=cmt_btcusdt"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| underlying\_index | String | Futures crypto |
| quote\_currency | String | Quote currency |
| coin | String | Margin token |
| contract\_val | String | Futures face value |
| delivery | Array | Settlement times |
| size\_increment | String | Decimal places of the quantity |
| tick\_size | String | Decimal places of the price |
| forwardContractFlag | Boolean | Whether it is USDT-M futures |
| priceEndStep | BigDecimal | Step size of the last decimal digit in the price |
| minLeverage | Integer | Minimum leverage (default: 1) |
| maxLeverage | Integer | Maximum leverage (default: 100) |
| buyLimitPriceRatio | String | Ratio of bid price to limit price |
| sellLimitPriceRatio | String | Ratio of ask price to limit price |
| makerFeeRate | String | Maker rate |
| takerFeeRate | String | Taker rate |
| minOrderSize | String | Minimum order size (base currency) |
| maxOrderSize | String | Maximum order size (base currency) |
| maxPositionSize | String | Maximum position size (base currency) |
| marketOpenLimitSize | String | Market Order Opening Position Single Limit (base currency) |

**Response example**

```json
[
  {
    "symbol": "cmt_btcusdt",
    "underlying_index": "BTC",
    "quote_currency": "USDT",
    "coin": "USDT",
    "contract_val": "0",
    "delivery": [
      "00:00:00",
      "08:00:00",
      "16:00:00"
    ],
    "size_increment": "5",
    "tick_size": "1",
    "forwardContractFlag": true,
    "priceEndStep": 1,
    "minLeverage": 1,
    "maxLeverage": 500,
    "buyLimitPriceRatio": "0.015",
    "sellLimitPriceRatio": "0.015",
    "makerFeeRate": "0.0002",
    "takerFeeRate": "0.0006",
    "minOrderSize": "0.0001",
    "maxOrderSize": "100000",
    "maxPositionSize": "1000000"
  }
]
```

- **GET** `/capi/v2/market/depth`

Weight(IP): 1

**Request parameters**

| Parameter | Parameter type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| limit | Integer | No | Fixed gear enumeration value: 15/200, the default gear is 15 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/depth?symbol=cmt_btcusdt&limit=15"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| asks | List | Sell side depth data   Format: `[price, quantity]` where quantity is in base currency |
| Index 0 | String | Price |
| Index 1 | String | Quantity |
| bids | List | Buy side depth data   Format: `[price, quantity]` where quantity is in base currency |
| Index 0 | String | Price |
| Index 1 | String | Quantity |
| timestamp | String | Timestamp   Unix millisecond timestamp |

**Response example**

```json
{
   "asks":[
         [
            "8858.0", //price
            "19299"//quantity
        ]   
     ],
   "bids":[
         [
            "7466.0", //price
            "499"  //quantity
        ],
         [
            "4995.0",
            "12500"
        ]
     ],
   "timestamp":"1591237821479" 
}
```

- **GET** `/capi/v2/market/tickers`

Weight(IP): 40

**Request parameters**

NONE

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/tickers"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| last | String | Latest execution price |
| best\_ask | String | Ask price |
| best\_bid | String | Bid price |
| high\_24h | String | Highest price in the last 24 hours |
| low\_24h | String | Lowest price in the last 24 hours |
| volume\_24h | String | Trading volume of quote currency |
| timestamp | String | System timestamp   Unix millisecond timestamp |
| priceChangePercent | String | Price increase or decrease (24 hours) |
| base\_volume | String | Trading volume of quote currency |
| markPrice | String | Mark price |
| indexPrice | String | Index price |

**Response example**

```json
[
  {
    "symbol": "cmt_btcusdt",
    "last": "90755.3",
    "best_ask": "90755.4",
    "best_bid": "90755.3",
    "high_24h": "91130.0",
    "low_24h": "90097.3",
    "volume_24h": "2321170547.37995",
    "timestamp": "1764482511864",
    "priceChangePercent": "0.000474",
    "base_volume": "25615.0755",
    "markPrice": "90755.2",
    "indexPrice": "90797.161"
  }
]
```

- **GET** `/capi/v2/market/ticker`

Weight(IP): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/ticker?symbol=cmt_btcusdt"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| last | String | Latest execution price |
| best\_ask | String | Ask price |
| best\_bid | String | Bid price |
| high\_24h | String | Highest price in the last 24 hours |
| low\_24h | String | Lowest price in the last 24 hours |
| volume\_24h | String | Trading volume of quote currency |
| timestamp | String | System timestamp   Unix millisecond timestamp |
| priceChangePercent | String | Price increase or decrease (24 hours) |
| base\_volume | String | Trading volume of quote currency |
| markPrice | String | Mark price |
| indexPrice | String | Index price |

**Response example**

```json
{
  "symbol": "cmt_btcusdt",
  "last": "90755.3",
  "best_ask": "90755.4",
  "best_bid": "90755.3",
  "high_24h": "91130.0",
  "low_24h": "90097.3",
  "volume_24h": "2321170547.37995",
  "timestamp": "1764482511864",
  "priceChangePercent": "0.000474",
  "base_volume": "25615.0755",
  "markPrice": "90755.2",
  "indexPrice": "90797.161"
}
```

- **GET** `/capi/v2/market/trades`

Weight(IP): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| limit | Integer | No | The size of the data ranges from 1 to 1000, with a default of 100 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/trades?symbol=cmt_btcusdt&limit=100"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| ticketId | String | Filled order ID |
| time | Long | The time at which the order was filled   Unix millisecond timestamp |
| price | String | The price at which the order was filled |
| size | String | The quantity that was filled (base currency) |
| value | String | Filled amount (quote currency) |
| symbol | String | Trading pair |
| isBestMatch | Boolean | Was the trade the best price match? |
| isBuyerMaker | Boolean | Was the buyer the maker? |
| contractVal | String | Futures face value (unit: contracts) |

**Response example**

```json
[
    {
        "ticketId": "124b129e-3999-4d14-a4b5-9bdda68e5e26",
        "time": 1716604853286,
        "price": "68734.8",
        "size": "0.001",
        "value": "68.7348",
        "symbol": "cmt_btcusdt",
        "isBestMatch": true,
        "isBuyerMaker": true,
        "contractVal": "0.000001"
    }
]
```

## Description

Query all historical K-line data and return a maximum of 100 pieces of data.  
When either startTime or endTime is invalid, the K-line data for the latest time period will be returned.  
When both startTime and endTime are provided, the endTime will take precedence and the startTime will be ignored.

- **GET** `/capi/v2/market/historyCandles`

Weight(IP): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| granularity | String | Yes | Candlestick interval\[1m,5m,15m,30m,1h,4h,12h,1d,1w\] |
| startTime | Long | No | The start time is to query the k-lines after this time   Unix millisecond timestamp |
| endTime | Long | No | The end time is to query the k-lines before this time   Unix millisecond timestamp |
| limit | Integer | No | The size of the data ranges from 1 to 100, with a default of 100 |
| priceType | String | No | Price Type: LAST latest market price; MARK mark; INDEX index;   LAST by default |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/historyCandles?symbol=cmt_bchusdt&granularity=1m"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| index\[0\] | String | Candlestick time   Unix millisecond timestamp |
| index\[1\] | String | Opening price |
| index\[2\] | String | Highest price |
| index\[3\] | String | Lowest price |
| index\[4\] | String | Closing price |
| index\[5\] | String | Trading volume of the base coin |
| index\[6\] | String | Trading volume of quote currency |

**Response example**

```json
[
  [
    "1716707460000",//Candlestick time
    "69174.3",//Opening price
    "69174.4",//Highest price
    "69174.1",//Lowest price
    "69174.3",//Closing price
    "0", //Trading volume of the base coin 
    "0.011" //Trading volume of quote currency
  ]
]
```

- **GET** `/capi/v2/market/candles`

Weight(IP): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| granularity | String | Yes | Candlestick interval\[1m,5m,15m,30m,1h,4h,12h,1d,1w\] |
| limit | Integer | No | The size of the data ranges from 1 to 1000, with a default of 100 |
| priceType | String | No | Price Type: LAST latest market price; MARK mark; INDEX index;   LAST by default |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/candles?symbol=cmt_bchusdt&granularity=1m"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| index\[0\] | String | Candlestick time   Unix millisecond timestamp |
| index\[1\] | String | Opening price |
| index\[2\] | String | Highest price |
| index\[3\] | String | Lowest price |
| index\[4\] | String | Closing price |
| index\[5\] | String | Trading volume of the base coin |
| index\[6\] | String | Trading volume of quote currency |

**Response example**

```json
[
    [
        "1716707460000",//Candlestick time
        "69174.3",//Opening price
        "69174.4",//Highest price
        "69174.1",//Lowest price
        "69174.3",//Closing price
        "0", //Trading volume of the base coin 
        "0.011" //Trading volume of quote currency
    ]
]
```

- **GET** `/capi/v2/market/index`

Weight(IP): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| priceType | String | No | Price Type: MARK mark; INDEX index;   INDEX by default |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/index?symbol=cmt_bchusdt&priceType=INDEX"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| index | String | Index |
| timestamp | String | Timestamp   Unix millisecond timestamp |

**Response example**

```json
{
    "symbol": "cmt_btcusdt",
    "index": "333.627857143",
    "timestamp": "1716604853286"
}
```

- **GET** `/capi/v2/market/open_interest`

Weight(IP): 2

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/open_interest?symbol=cmt_1000satsusdt"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| base\_volume | String | Total open interest of the platform Specific coins |
| target\_volume | String | Quote Currency Holdings |
| timestamp | String | Timestamp   Unix millisecond timestamp |

**Response example**

```json
[
    {
        "symbol": "cmt_1000satsusdt",
        "base_volume": "0",
        "target_volume": "0",
        "timestamp": "1716709712753"
    }
]
```

Weight(IP): 1

- **GET** `/capi/v2/market/funding_time`

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/funding_time?symbol=cmt_bchusdt"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| fundingTime | Long | Settlement time   Unix millisecond timestamp |

**Response example**

```json
{
  "symbol": "cmt_btcusdt",
  "fundingTime": 1716595200000
}
```

- **GET** `/capi/v2/market/getHistoryFundRate`

Weight(IP): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| limit | Integer | No | The size of the data ranges from 1 to 100, with a default of 10 |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/getHistoryFundRate?symbol=cmt_bchusdt&limit=100"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | string | Trading pair |
| fundingRate | string | Funding rate |
| fundingTime | long | Funding fee settlement time   Unix millisecond timestamp |

**Response example**

```json
[
    {
        "symbol": "cmt_btcusdt",
        "fundingRate": "0.0001028",
        "fundingTime": 1716595200000
    }
]
```

- **GET** `/capi/v2/market/currentFundRate`

Weight(IP): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/market/currentFundRate?symbol=cmt_bchusdt"
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| symbol | String | Trading pair |
| fundingRate | String | Current funding rates |
| collectCycle | Long | Funding rate settlement cycle   Unit: minute |
| timestamp | Long | Funding fee settlement time   Unix millisecond timestamp |

**Response example**

```json
[
  {
    "symbol": "cmt_btcusdt",
    "fundingRate": "-0.0001036",
    "collectCycle": 480,
    "timestamp": 1750383726052
  }
]
```