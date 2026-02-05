- **GET** `/capi/v2/account/getAccounts`

**Required permission:** Futures trading read permissions

Weight(IP): 5, Weight(UID): 5

**Request parameters**

NONE

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/getAccounts" \
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
| account | Object | Account information |
| \> defaultFeeSetting | Object | Default fee configuration |
| \>> is\_set\_fee\_rate | Boolean | Whether fee rates are set |
| \>> taker\_fee\_rate | String | Taker fee rate |
| \>> maker\_fee\_rate | String | Maker fee rate |
| \>> is\_set\_fee\_discount | Boolean | Whether fee discounts are enabled |
| \>> fee\_discount | String | Account transaction fee discounts |
| \>> is\_set\_taker\_maker\_fee\_discount | Boolean | Whether to apply separate fee discounts for takers and makers |
| \>> taker\_fee\_discount | String | Taker fee rate discount |
| \>> maker\_fee\_discount | String | Maker fee rate discount |
| \> feeSetting | Array Object | Fee settings |
| \>> symbol | String | Symbol name |
| \>> is\_set\_fee\_rate | Boolean | Whether fee rates are set |
| \>> taker\_fee\_rate | String | Taker fee rate |
| \>> maker\_fee\_rate | String | Maker fee rate |
| \>> is\_set\_fee\_discount | Boolean | Whether fee discounts are enabled |
| \>> fee\_discount | String | Fee rate discount |
| \>> is\_set\_taker\_maker\_fee\_discount | Boolean | Whether to apply separate fee discounts for takers and makers |
| \>> taker\_fee\_discount | String | Taker fee rate discount |
| \>> maker\_fee\_discount | String | Maker fee rate discount |
| \> modeSetting | Array Object | Mode settings |
| \>> symbol | String | Symbol name |
| \>> marginMode | String | Margin mode |
| \>> separatedMode | String | Position segregation mode |
| \>> positionMode | String | Position mode |
| \> leverageSetting | Array Object | Leverage settings |
| \>> symbol | String | Symbol name |
| \>> isolated\_long\_leverage | String | Isolated long position leverage |
| \>> isolated\_short\_leverage | String | Isolated short position leverage |
| \>> cross\_leverage | String | Cross margin leverage |
| \> createOrderRateLimitPerMinute | Integer | Order creation rate limit per minute |
| \> createOrderDelayMilliseconds | Integer | Order creation delay (milliseconds) |
| \> createdTime | String | Creation time   Unix millisecond timestamp |
| \> updatedTime | String | Update time   Unix millisecond timestamp |
| collateral | Array Object | Collateral information |
| \> coin | String | Currency |
| \> marginMode | String | Margin mode |
| \> crossSymbol | String | When marginMode=CROSS, represents the symbol associated with cross margin mode. Null in other cases. |
| \> isolated\_position\_id | String | When marginMode=ISOLATED, represents the position ID associated with isolated margin. 0 in other cases. |
| \> amount | String | Collateral amount |
| \> pending\_deposit\_amount | String | Pending deposit amount |
| \> pending\_withdraw\_amount | String | Pending withdrawal amount |
| \> pending\_transfer\_in\_amount | String | Pending inbound transfer amount |
| \> pending\_transfer\_out\_amount | String | Pending outbound transfer amount |
| \> is\_liquidating | Boolean | Whether liquidation is triggered (in progress) |
| \> legacy\_amount | String | Legacy balance (display only) |
| \> cum\_deposit\_amount | String | Accumulated deposit amount |
| \> cum\_withdraw\_amount | String | Accumulated withdrawal amount |
| \> cum\_transfer\_in\_amount | String | Accumulated inbound transfer amount |
| \> cum\_transfer\_out\_amount | String | Accumulated outbound transfer amount |
| \> cum\_margin\_move\_in\_amount | String | Accumulated margin deposit amount |
| \> cum\_margin\_move\_out\_amount | String | Accumulated margin withdrawal amount |
| \> cum\_position\_open\_long\_amount | String | Accumulated collateral amount for long position openings |
| \> cum\_position\_open\_short\_amount | String | Accumulated collateral amount for short position openings |
| \> cum\_position\_close\_long\_amount | String | Accumulated collateral amount for long position closings |
| \> cum\_position\_close\_short\_amount | String | Accumulated collateral amount for short position closings |
| \> cum\_position\_fill\_fee\_amount | String | Accumulated trading fees for filled orders |
| \> cum\_position\_liquidate\_fee\_amount | String | Accumulated liquidation fees |
| \> cum\_position\_funding\_amount | String | Accumulated funding fees |
| \> cum\_order\_fill\_fee\_income\_amount | String | Accumulated order fee income |
| \> cum\_order\_liquidate\_fee\_income\_amount | String | Accumulated liquidation fee income |
| \> created\_time | String | Creation time,Unix millisecond timestamp |
| \> updated\_time | String | Update time,Unix millisecond timestamp |
| version | String | Version number |

**Response example**

```json
{
  "account": {
    "defaultFeeSetting": {
      "is_set_fee_rate": true,
      "taker_fee_rate": "0.00072000",
      "maker_fee_rate": "0.00018000",
      "is_set_fee_discount": false,
      "fee_discount": "0",
      "is_set_taker_maker_fee_discount": false,
      "taker_fee_discount": "0",
      "maker_fee_discount": "0"
    },
    "feeSetting": [
      {
        "symbol": "cmt_btcusdt",
        "is_set_fee_rate": false,
        "taker_fee_rate": "0.00072000",
        "maker_fee_rate": "0.00018000",
        "is_set_fee_discount": false,
        "fee_discount": "0",
        "is_set_taker_maker_fee_discount": false,
        "taker_fee_discount": "0",
        "maker_fee_discount": "0"
      }
    ],
    "modeSetting": [
      {
        "symbol": "cmt_btcusdt",
        "marginMode": "SHARED",
        "separatedMode": "COMBINED",
        "positionMode": "HEDGE"
      }
    ],
    "leverageSetting": [
      {
        "symbol": "cmt_btcusdt",
        "isolated_long_leverage": "1",
        "isolated_short_leverage": "100",
        "cross_leverage": "20.00"
      }
    ],
    "createOrderRateLimitPerMinute": 0,
    "createOrderDelayMilliseconds": 0,
    "createdTime": 1728493655673,
    "updatedTime": 1764221456649
  },
  "collateral": [
    {
      "coin": "USDT",
      "marginMode": "SHARED",
      "crossSymbol": null,
      "isolatedPositionId": 0,
      "amount": "4663.20125463",
      "pending_deposit_amount": "0.00000000",
      "pending_withdraw_amount": "0.000000",
      "pending_transfer_in_amount": "0",
      "pending_transfer_out_amount": "0.00000000",
      "is_liquidating": false,
      "legacy_amount": "5662.90202073",
      "cum_deposit_amount": "6860279.70836100",
      "cum_withdraw_amount": "6647240.391794",
      "cum_transfer_in_amount": "452.09021000",
      "cum_transfer_out_amount": "292.75030000",
      "cum_margin_move_in_amount": "10208.01422763",
      "cum_margin_move_out_amount": "12366.05275584",
      "cum_position_open_long_amount": "4672962.7674221",
      "cum_position_open_short_amount": "287777.669958",
      "cum_position_close_long_amount": "4532201.78518940",
      "cum_position_close_short_amount": "287888.538892",
      "cum_position_fill_fee_amount": "0.59982046",
      "cum_position_liquidate_fee_amount": "76.930498",
      "cum_position_funding_amount": "9859.37164167",
      "cum_order_fill_fee_income_amount": "0",
      "cum_order_liquidate_fee_income_amount": "0",
      "created_time": 1728493664997,
      "updated_time": 1764306706243
    }
  ],
  "version": null
}
```

- **GET** `/capi/v2/account/getAccount`

Weight(IP): 1, Weight(UID): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| coin | String | Yes | coin name |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/getAccount?coin=USDT" \
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
| account | Object | Account information |
| \> defaultFeeSetting | Object | Default fee configuration |
| \>> is\_set\_fee\_rate | Boolean | Whether fee rates are set |
| \>> taker\_fee\_rate | String | Taker fee rate |
| \>> maker\_fee\_rate | String | Maker fee rate |
| \>> is\_set\_fee\_discount | Boolean | Whether fee discounts are enabled |
| \>> fee\_discount | String | Fee rate discount |
| \>> is\_set\_taker\_maker\_fee\_discount | Boolean | Whether to apply separate fee discounts for takers and makers |
| \>> taker\_fee\_discount | String | Taker fee rate discount |
| \>> maker\_fee\_discount | String | Maker fee rate discount |
| \> feeSetting | Array Object | Fee settings |
| \>> symbol | String | Symbol name |
| \>> is\_set\_fee\_rate | Boolean | Whether fee rates are set |
| \>> taker\_fee\_rate | String | Taker fee rate |
| \>> maker\_fee\_rate | String | Maker fee rate |
| \>> is\_set\_fee\_discount | Boolean | Whether fee discounts are enabled |
| \>> fee\_discount | String | Fee rate discount |
| \>> is\_set\_taker\_maker\_fee\_discount | Boolean | Whether to apply separate fee discounts for takers and makers |
| \>> taker\_fee\_discount | String | Taker fee rate discount |
| \>> maker\_fee\_discount | String | Maker fee rate discount |
| \> modeSetting | Array Object | Mode settings |
| \>> symbol | String | Symbol name |
| \>> marginMode | String | Margin mode |
| \>> separated\_mode | String | Position segregation mode |
| \>> position\_mode | String | Position mode |
| \> leverageSetting | Array Object | Leverage settings |
| \>> symbol | String | Symbol name |
| \>> isolated\_long\_leverage | String | Isolated long position leverage |
| \>> isolated\_short\_leverage | String | Isolated short position leverage |
| \>> cross\_leverage | String | Cross margin leverage |
| \> createOrderRateLimitPerMinute | Integer | Order creation rate limit per minute |
| \> createOrderDelayMilliseconds | Integer | Order creation delay (milliseconds) |
| \> createdTime | String | Creation time   Unix millisecond timestamp |
| \> updatedTime | String | Update time   Unix millisecond timestamp |
| collateral | Array Object | Collateral information |
| \> coin | String | Currency |
| \> marginMode | String | Margin mode |
| \> crossSymbol | String | When marginMode=CROSS, represents the symbol associated with cross margin mode. Null in other cases. |
| \> isolatedPositionId | String | When marginMode=ISOLATED, represents the position ID associated with isolated margin. 0 in other cases. |
| \> amount | String | Collateral amount |
| \> pending\_deposit\_amount | String | Pending deposit amount |
| \> pending\_withdraw\_amount | String | Pending withdrawal amount |
| \> pending\_transfer\_in\_amount | String | Pending inbound transfer amount |
| \> pending\_transfer\_out\_amount | String | Pending outbound transfer amount |
| \> is\_liquidating | Boolean | Whether liquidation is triggered (in progress) |
| \> legacy\_amount | String | Legacy balance (display only) |
| \> cum\_deposit\_amount | String | Accumulated deposit amount |
| \> cum\_withdraw\_amount | String | Accumulated withdrawal amount |
| \> cum\_transfer\_in\_amount | String | Accumulated inbound transfer amount |
| \> cum\_transfer\_out\_amount | String | Accumulated outbound transfer amount |
| \> cum\_margin\_move\_in\_amount | String | Accumulated margin deposit amount |
| \> cum\_margin\_move\_out\_amount | String | Accumulated margin withdrawal amount |
| \> cum\_position\_open\_long\_amount | String | Accumulated collateral amount for long position openings |
| \> cum\_position\_open\_short\_amount | String | Accumulated collateral amount for short position openings |
| \> cum\_position\_close\_long\_amount | String | Accumulated collateral amount for long position closings |
| \> cum\_position\_close\_short\_amount | String | Accumulated collateral amount for short position closings |
| \> cum\_position\_fill\_fee\_amount | String | Accumulated trading fees for filled orders |
| \> cum\_position\_liquidate\_fee\_amount | String | Accumulated liquidation fees |
| \> cum\_position\_funding\_amount | String | Accumulated funding fees |
| \> cum\_order\_fill\_fee\_income\_amount | String | Accumulated order fee income |
| \> cum\_order\_liquidate\_fee\_income\_amount | String | Accumulated liquidation fee income |
| \> created\_time | String | Creation time   Unix millisecond timestamp |
| \> updated\_time | String | Update time   Unix millisecond timestamp |
| version | String | Version number |

**Response example**

```json
{
  "account": {
    "defaultFeeSetting": {
      "is_set_fee_rate": true,
      "taker_fee_rate": "0.00072000",
      "maker_fee_rate": "0.00018000",
      "is_set_fee_discount": false,
      "fee_discount": "0",
      "is_set_taker_maker_fee_discount": false,
      "taker_fee_discount": "0",
      "maker_fee_discount": "0"
    },
    "feeSetting": [
      {
        "symbol": "cmt_btcusdt",
        "is_set_fee_rate": false,
        "taker_fee_rate": "0.00078000",
        "maker_fee_rate": "0.00018000",
        "is_set_fee_discount": false,
        "fee_discount": "0",
        "is_set_taker_maker_fee_discount": false,
        "taker_fee_discount": "0",
        "maker_fee_discount": "0"
      }
    ],
    "modeSetting": [
      {
        "symbol": "cmt_btcusdt",
        "marginMode": "SHARED",
        "separatedModeEnum": "COMBINED",
        "positionModeEnum": "HEDGE"
      }
    ],
    "leverageSetting": [
      {
        "symbol": "cmt_btcusdt",
        "isolated_long_leverage": "1",
        "isolated_short_leverage": "100",
        "cross_leverage": "20.00"
      }
    ],
    "createOrderRateLimitPerMinute": 0,
    "createOrderDelayMilliseconds": 0,
    "createdTime": 1728493655673,
    "updatedTime": 1764221456649
  },
  "collateral": [
    {
      "coin": "USDT",
      "marginMode": "SHARED",
      "crossSymbol": null,
      "isolatedPositionId": 0,
      "amount": "4663.20125463",
      "pending_deposit_amount": "0.00000000",
      "pending_withdraw_amount": "0.000000",
      "pending_transfer_in_amount": "0",
      "pending_transfer_out_amount": "0.00000000",
      "is_liquidating": false,
      "legacy_amount": "5662.90202073",
      "cum_deposit_amount": "6860279.70836100",
      "cum_withdraw_amount": "6647240.391794",
      "cum_transfer_in_amount": "452.09021000",
      "cum_transfer_out_amount": "292.75030000",
      "cum_margin_move_in_amount": "10208.01422763",
      "cum_margin_move_out_amount": "12366.05275584",
      "cum_position_open_long_amount": "4672962.7674221",
      "cum_position_open_short_amount": "287777.669958",
      "cum_position_close_long_amount": "4532201.78518940",
      "cum_position_close_short_amount": "287888.538892",
      "cum_position_fill_fee_amount": "0.59982046",
      "cum_position_liquidate_fee_amount": "76.930498",
      "cum_position_funding_amount": "9859.37164167",
      "cum_order_fill_fee_income_amount": "0",
      "cum_order_liquidate_fee_income_amount": "0",
      "created_time": 1728493664997,
      "updated_time": 1764306706243
    }
  ],
  "version": null
}
```

**HTTP request** Retrieve account assets

- **GET** `/capi/v2/account/assets`

Weight(IP): 5, Weight(UID): 10

**Request parameters**

NONE

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/assets" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json"
```

**Response parameters**

| Field Name | Type | Field Description |
| --- | --- | --- |
| coinName | String | Name of the crypto |
| available | String | Available asset |
| frozen | String | Frozen asset |
| equity | String | Total asset |
| unrealizePnl | String | Unrealized Profit and Loss |

**Response example**

```json
[
  {
    "coinName": "USDT",
    "available": "5413.06877369",
    "equity": "5696.49288823",
    "frozen": "81.28240000",
    "unrealizePnl": "-34.55300000"
  }
]
```

**HTTP request** Get Contract Account Bill History

- **POST** `/capi/v2/account/bills`

Weight(IP): 2, Weight(UID): 5

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| coin | String | No | Currency name |
| symbol | String | No | Trading pair |
| businessType | String | No | Business type   deposit: Deposit   withdraw: Withdrawal   transfer\_in: Transfer between different accounts (in)   transfer\_out: Transfer between different accounts (out)   margin\_move\_in: Collateral transferred within the same account due to opening/closing positions, manual/auto addition   margin\_move\_out: Collateral transferred out within the same account due to opening/closing positions, manual/auto addition   position\_open\_long: Collateral change from opening long positions (buying decreases collateral)   position\_open\_short: Collateral change from opening short positions (selling increases collateral)   position\_close\_long: Collateral change from closing long positions (selling increases collateral)   position\_close\_short: Collateral change from closing short positions (buying decreases collateral)   position\_funding: Collateral change from position funding fee settlement   order\_fill\_fee\_income: Order fill fee income (specific to fee account)   order\_liquidate\_fee\_income: Order liquidation fee income (specific to fee account)   start\_liquidate: Start liquidation   finish\_liquidate: Finish liquidation   order\_fix\_margin\_amount: Compensation for liquidation loss   tracking\_follow\_pay: Copy trading payment, pre-deducted from followers after position closing if profitable   tracking\_system\_pre\_receive: Pre-received commission, commission system account receives pre-deducted amount from followers   tracking\_follow\_back: Copy trading commission refund   tracking\_trader\_income: Lead trader income   tracking\_third\_party\_share: Profit sharing (shared by lead trader with others) |
| startTime | Long | No | Start timestamp   Unit: milliseconds |
| endTime | Long | No | End timestamp   Unit: milliseconds |
| limit | Integer | No | Return record limit, default: 20   Minimum: 1   Maximum: 100 |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/account/bills" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*******" \
   -H "ACCESS-PASSPHRASE:*****" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{
    "coin": "",
    "symbol": "",
    "businessType": "",
    "startTime": null,
    "endTime": null,
    "limit":10}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| billId | String | Bill ID |
| coin | String | Currency name |
| symbol | String | Trading pair |
| amount | String | Amount |
| businessType | String | Transaction business type |
| balance | String | Balance |
| fillFee | String | Transaction fee |
| transferReason | String | Transfer Reason   UNKNOWN\_TRANSFER\_REASON: Unknown transfer reason   USER\_TRANSFER: User manual transfer   INCREASE\_CONTRACT\_CASH\_GIFT: Increase contract cash gift   REDUCE\_CONTRACT\_CASH\_GIFT: Reduce contract cash gift   REFUND\_WXB\_DISCOUNT\_FEE: Refund WXB discount fee |
| cTime | String | Creation time   Unix millisecond timestamp |

**Response example**

```json
{
  "hasNextPage": true,
  "items": [
    {
      "billId": 686960019383517338,
      "coin": "USDT",
      "symbol": "cmt_btcusdt",
      "amount": "0.08266646",
      "businessType": "position_funding",
      "balance": "4738.70667369",
      "fillFee": "0",
      "transferReason": "UNKNOWN_TRANSFER_REASON",
      "ctime": 1763784031721
    }
  ]
}
```

**Required permission:** Futures account read permissions

- **GET** `/capi/v2/account/settings`

Weight(IP): 1, Weight(UID): 1

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | No | Trading pair   If not filled in, all will be returned by default. |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/settings?symbol=cmt_bchusdt" \
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
| symbol | object | Trading pair |
| \> isolated\_long\_leverage | string | Isolated long position leverage |
| \> isolated\_short\_leverage | string | Isolated short position leverage |
| \> cross\_leverage | string | Cross margin leverage |

**Response example**

```json
{
  "cmt_ethusdt": {
    "isolated_long_leverage": "20.00",
    "isolated_short_leverage": "20.00",
    "cross_leverage": "20.00"
  }
}
```

- **POST** `/capi/v2/account/leverage`

Weight(IP): 10, Weight(UID): 20

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| marginMode | Integer | Yes | Margin mode   1: Cross Mode   3: Isolated Mode   The marginMode must be set to the account's current mode. |
| longLeverage | String | Yes | Long position leverage   In Cross Mode, must be identical to shortLeverage. |
| shortLeverage | String | Yes | Short position leverage   In Cross Mode, must be identical to longLeverage. |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/account/leverage" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol":"cmt_bchusdt","marginMode":1,"longLeverage":"2"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| msg | string | Response message |
| requestTime | string | Timestamp   Unix millisecond timestamp |
| code | string | Response code |

**Response example**

```json
{
    "msg": "success",
    "requestTime": 1713339011237,
    "code": "200"
}
```

- **POST** `/capi/v2/account/adjustMargin`

Weight(IP): 15, Weight(UID): 30

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| coinId | Integer | No | Collateral ID [Basic Crypto Information](https://www.weex.com/api-doc/spot/ConfigAPI/CurrencyInfo)   Default is 2 (USDT) |
| isolatedPositionId | Long | Yes | Isolated margin position ID   [Get the isolatedPositionId](https://www.weex.com/api-doc/contract/Account_API/GetSingleContractPosition) |
| collateralAmount | String | Yes | Collateral amount   positive means increase, and negative means decrease |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/account/adjustMargi" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"coinId":2,"isolatedPositionId":1,"collateralAmount":"10"}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| msg | String | Response message |
| requestTime | String | Timestamp   Unix millisecond timestamp |
| code | String | Response code |

**Response example**

```json
{
    "msg": "success",
    "requestTime": 1713339011237,
    "code": "200"
}
```

- **POST** `/capi/v2/account/modifyAutoAppendMargin`

Weight(IP): 15, Weight(UID): 30

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| positionId | Long | Yes | Isolated margin position ID |
| autoAppendMargin | Boolean | Yes | Whether to enable automatic margin call |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/account/modifyAutoAppendMargin" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"positionId":1,"autoAppendMargin":false}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| msg | String | Response message |
| requestTime | String | Timestamp   Unix millisecond timestamp |
| code | String | Response code |

**Response example**

```json
{
    "msg": "success",
    "requestTime": 1713339011237,
    "code": "200"
}
```

- **GET** `/capi/v2/account/position/allPosition`

Weight(IP): 10, Weight(UID): 15

**Request parameters**

NONE

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/position/allPosition" \
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
| id | Long | Position ID |
| account\_id | Long | Associated account ID |
| coin\_id | Integer | Associated collateral currency ID |
| contract\_id | Long | Associated futures ID |
| symbol | String | Trading pair |
| side | String | Position direction such as LONG or SHORT |
| margin\_mode | String | Margin mode of current position   SHARED: Cross Mode   ISOLATED: Isolated Mode |
| separated\_mode | String | Current position's separated mode   COMBINED: Combined mode   SEPARATED: Separated mode |
| separated\_open\_order\_id | Long | Opening order ID of separated position |
| leverage | String | Position leverage |
| size | String | Current position size |
| open\_value | String | Initial value at position opening |
| open\_fee | String | Opening fee |
| funding\_fee | String | Funding fee |
| marginSize | String | Margin amount (margin coin) |
| isolated\_margin | String | Isolated margin |
| is\_auto\_append\_isolated\_margin | boolean | Whether the auto-adding of funds for the isolated margin is enabled (only for isolated mode) |
| cum\_open\_size | String | Accumulated opened positions |
| cum\_open\_value | String | Accumulated value of opened positions |
| cum\_open\_fee | String | Accumulated fees paid for opened positions |
| cum\_close\_size | String | Accumulated closed positions |
| cum\_close\_value | String | Accumulated value of closed positions |
| cum\_close\_fee | String | Accumulated fees paid for closing positions |
| cum\_funding\_fee | String | Accumulated settled funding fees |
| cum\_liquidate\_fee | String | Accumulated liquidation fees |
| created\_match\_sequence\_id | Long | Matching engine sequence ID at creation |
| updated\_match\_sequence\_id | Long | Matching engine sequence ID at last update |
| created\_time | Long | Creation time   Unix millisecond timestamp |
| updated\_time | Long | Update time   Unix millisecond timestamp |
| contractVal | String | Futures face value |
| unrealizePnl | String | Unrealized PnL |
| liquidatePrice | String | Estimated liquidation price   If the value = 0, it means the position is at low risk and there is no liquidation price at this time |

**Response example**

- **GET** `/capi/v2/account/position/singlePosition`

Weight(IP): 2, Weight(UID): 3

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |

**Request example**

```powershell
curl "https://api-contract.weex.com/capi/v2/account/position/singlePosition?symbol=cmt_bchusdt" \
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
| id | Long | Position ID |
| account\_id | Long | Associated account ID |
| coin\_id | Integer | Associated collateral currency ID |
| contract\_id | Long | Associated futures ID |
| symbol | String | Trading pair |
| side | String | Position direction such as LONG or SHORT |
| margin\_mode | String | Margin mode of current position   SHARED: Cross Mode   ISOLATED: Isolated Mode |
| separated\_mode | String | Current position's separated mode   COMBINED: Combined mode   SEPARATED: Separated mode |
| separated\_open\_order\_id | Long | Opening order ID of separated position |
| leverage | String | Position leverage |
| size | String | Current position size |
| open\_value | String | Initial value at position opening |
| open\_fee | String | Opening fee |
| funding\_fee | String | Funding fee |
| marginSize | String | Margin amount (margin coin) |
| isolated\_margin | String | Isolated margin |
| is\_auto\_append\_isolated\_margin | boolean | Whether the auto-adding of funds for the isolated margin is enabled (only for isolated mode) |
| cum\_open\_size | String | Accumulated opened positions |
| cum\_open\_value | String | Accumulated value of opened positions |
| cum\_open\_fee | String | Accumulated fees paid for opened positions |
| cum\_close\_size | String | Accumulated closed positions |
| cum\_close\_value | String | Accumulated value of closed positions |
| cum\_close\_fee | String | Accumulated fees paid for closing positions |
| cum\_funding\_fee | String | Accumulated settled funding fees |
| cum\_liquidate\_fee | String | Accumulated liquidation fees |
| created\_match\_sequence\_id | Long | Matching engine sequence ID at creation |
| updated\_match\_sequence\_id | Long | Matching engine sequence ID at last update |
| created\_time | Long | Creation time   Unix millisecond timestamp |
| updated\_time | Long | Update time   Unix millisecond timestamp |
| contractVal | String | Futures face value |
| unrealizePnl | String | Unrealized PnL |
| liquidatePrice | String | Estimated liquidation price   If the value = 0, it means the position is at low risk and there is no liquidation price at this time |

**Response example**

- **POST** `/capi/v2/account/position/changeHoldModel`

Weight(IP): 20, Weight(UID): 50

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| symbol | String | Yes | Trading pair |
| marginMode | Integer | Yes | Margin mode   1: Cross Mode   3: Isolated Mode |
| separatedMode | Integer | No | Position segregation mode   1: Combined mode   System will automatically set to Combined mode by default |

**Request example**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/account/position/changeHoldModel" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{"symbol":"cmt_bchusdt","marginMode":1}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| msg | string | Response message |
| requestTime | string | Timestamp   Unix millisecond timestamp |
| code | string | Response code |

**Response example**

```json
{
    "msg": "success",
    "requestTime": 1713339011237,
    "code": "200"
}
```