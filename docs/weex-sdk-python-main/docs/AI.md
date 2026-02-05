1. All trading must be conducted through the official WEEX OpenAPI platform using accounts approved for the competition.
2. Only the designated trading pairs announced for the competition are permitted for trading.`cmt_btcusdt`, `cmt_ethusdt`, `cmt_solusdt`, `cmt_dogeusdt`, `cmt_xrpusdt`, `cmt_adausdt`, `cmt_bnbusdt`,`cmt_ltcusdt`
3. Any trades outside the approved scope will result in immediate disqualification.
4. Trading strategies may be fully or semi-automated, but purely manual trading is strictly prohibited.
5. The following actions are prohibited and will lead to disqualification:
- Profiting from high-frequency or latency arbitrage, wash trading, or market manipulation.
- Altering data, falsifying trading records, or fabricating AI logs.
- Sharing, borrowing, or transferring API keys or trading accounts.
- Engaging in purely manual trading without the use of genuine AI technology.
1. Any participant found attempting to gain an unfair advantage through deceptive or manipulative means will be disqualified, and their ranking will be voided.
2. The organizer reserves the right to implement or adjust risk control measures (including maximum leverage, position limits, order frequency limits, STP, rate limiting, and liquidation protection mechanisms) and may enforce position closures to manage risk or comply with legal and regulatory requirements.

## Welcome to the Arena: The Path to Alpha Awakening

**AI Wars: WEEX Alpha Awakens – Global AI Trading Hackathon!**

In this ultimate showdown, top developers, quants, and traders from around the world will unleash their algorithms in real-market battles, competing for one of the richest prize pools in AI crypto trading history: **880,000 USD**, including a **Bentley Bentayga S** for the champion.

This guide will walk you through every required step from registration to the official start of the competition.

Follow the path and start your journey:

**Register & Form Your Team → Pass API Testing → Model Tuning → Official Start**

Check the registration tutorial: [https://www.youtube.com/watch?v=yeJaQRN9spA](https://www.youtube.com/watch?v=yeJaQRN9spA)

**Goal:** Complete your official registration, create or join a team (BUIDL), and pass the review to receive your dedicated API key.

**Outcome:** You will obtain exclusive API credentials to connect your system to WEEX — marking your first step into the competition.

### 1.1 Visit the AI Wars: WEEX Alpha Awakens Event Page

1. Visit the event page: [https://www.weex.com/events/ai-trading](https://www.weex.com/events/ai-trading)
2. Find the **"Submit BUIDL"** button and click

![img.png](https://www.weex.com/api-doc/assets/images/img-c08e772b4873c916718c8d930d9bf5bb.png)

### 1.2 Find a Team or Build Your Own

A **BUIDL** is the basic participating unit of this competition and represents a team.

Existing BUIDLs represent teams that have already been created. You may join one of these BUIDLs or create your own team from scratch.

![img.png](https://www.weex.com/api-doc/assets/images/img1-f9593022a1d1701509eca6d1af58b319.png)

If you choose to submit your own BUIDL, information should include:

- Profile: BUIDL name, logo, vision, category, GitHub (optional), and social links
- Details: A brief introduction to your BUIDL
- Team: Team information. You can also invite or recruit team members to this section.
- Contact: Telegram handle and backup contact details
- Submission：
	- WEEX UID (KYC required)
	- IP Address (this IP will be added to the WEEX OpenAPI whitelist to enable successful API calls)
	- Preferred programming languages
	- Experience with Large Language Models (LLMs), AI-assisted trading, automated trading bots, or other exchanges’ APIs
	- Number of orders your strategy will place per day

#### 1: How to Complete KYC and Find Your WEEX UID

To register for the WEEX Global AI Trading Hackathon, you’ll need to provide your KYC-verified WEEX UID. Here’s how to find it:

1. **Register Your Account**
	Click the [link](https://www.weex.com/) to visit the WEEX official website. Select “Sign Up” in the top right, then register using your email or phone number.
	**Note:** If you already have an account, click “Log In” in the top-right corner to access your dashboard.

![img.png](https://www.weex.com/api-doc/assets/images/img2-cc6d2aea5f50df9432d13f4354fccbff.png)

1. **Complete Identity Verification**
	Click the avatar icon in the top right and select “Verification” to complete your KYC.
	**Note:** KYC is mandatory — submissions without KYC cannot be approved.

![img.png](https://www.weex.com/api-doc/assets/images/img5-836885e27a1710d0a0afffc8468e41b7.png)

1. **Find Your UID**
	Click the avatar icon again, and you will see your UID displayed right below your email.

![img_1.png](https://www.weex.com/api-doc/assets/images/img4-ea2b5e96a2d8bfc18846850f9335abe0.png)

#### Submission Mini Tip 2: How to Find Your IP Address

**Part 1: The Recommended Method (Cloud Servers)**

For best stability, we strongly recommend using a cloud server with static public IP and supporting **24/7 uninterrupted operation such as:** AWS (Amazon Web Services), Alibaba Cloud, and Tencent Cloud.

**Part 2: The Alternative Method (Local Computer)**

If you choose to run your trading bot from a personal computer or home network, you must confirm that your outbound IP address is static. A changing IP will result in connectivity issues.

You have two main options to ensure a stable outbound IP:

1. **Use a static IP** provided by your Internet Service Provider (ISP).
2. **Use a VPN or Proxy service** with a fixed egress IP (and ensure the VPN/Proxy is **consistently enabled** without switching servers).

**Steps to find your local public IP:**

- **Turn off all VPNs**, or keep only the single VPN whose IP you plan to whitelist.
- Visit [whatismyip.com](http://whatismyip.com/) in your browser.
- The page will show your **public IPv4 address**.
- Copy this IP and submit it to the whitelist.

### 1.3 Missing Information? We Will Follow Up

After you submit your BUIDL, the WEEX team will review your application based on the competition requirements. The review process normally takes **one business day**.

If any information is missing or requires clarification, our team will reach out to you through one of the following channels:

- DoraHacks messaging system
- WEEX official messaging system
- Your registered contact information (Telegram, X, etc.)

Please keep your contact details active and accessible.

Once your BUIDL is approved, you will receive your **competition account** and **exclusive API Key**, which will allow you to move on to the next stage: **API testing and model integration.**

| ![img_2.png](https://www.weex.com/api-doc/assets/images/img_2-71304ffbb29dfaa671974f922ba816e3.png) | ![img_3.png](https://www.weex.com/api-doc/assets/images/img_3-61fa86b491e9e40aefbe2d00e31f1968.png) |
| --- | --- |
| Under Review | Approved |

### 1.4 Your Starter Kit

After your BUIDL passes the review, WEEX will create a dedicated competition account for you and provide the API credentials and testing information required for the next stage. These details will be sent to you via the DoraHacks message system and WEEX Labs official emails.

You will receive:

- **API Key:** The identifier for your competition account and a required parameter for all API requests.
- **Secret Key:** System-generated key used for request signing and security verification.
- **Passphrase:** Required to perform API operations.
- **API Testing Page Link:** A mini testing environment where you can view the API testing requirements, specifications, and completion criteria.

**With this, your registration is complete and you’re ready for API testing.**

## Step 2｜Pass the Gateway: Complete Your API Testing

**Goal:** Ensure that your system can successfully interact with the WEEX API and execute the required test trades.

**Outcome:** You’ll secure official entry qualification, receive initial test funds for debugging, and gain a clear understanding of all pre-competition requirements.

### 2.1 Instructions

- Please complete all required operations listed on the API Testing page.
- Participants who complete and pass the API testing will officially obtain eligibility for the competition.
- Participants who fail to complete or pass API testing will be unable to proceed to the model integration and formal competition stages.

After receiving the above information, please keep your API credentials secure and follow the instructions to complete the required tests.

![img_4.png](https://www.weex.com/api-doc/assets/images/img_4-43fadc11502f04768312588272f0d3ce.png)

### 2.2 Connect and Test

Participants need to complete a simple API test to qualify for the preliminary round of the hackathon. Please read the WEEX official API documentation carefully, and use the API key we provide to complete the test.

#### Integration Preparation

1. Please read the official WEEX API documentation carefully: [https://www.weex.com/api-doc/ai/intro](https://www.weex.com/api-doc/ai/intro)
2. Connect to a cloud server and run the code below. You should receive a response that confirms whether your network connection is working properly.
```powershell
curl -s --max-time 10 "https://api-contract.weex.com/capi/v2/market/time"
```
```json
{"epoch":"1765423487.896","iso":"2025-12-11T03:24:47.896Z","timestamp":1765423487896}
```
1. If your project is developed in **Java or Python**, you can directly use the corresponding code examples provided in the documentation. For other programming languages, please adapt the examples according to the official API documentation to suit your implementation. Sample code documentation: [https://www.weex.com/api-doc/ai/QuickStart/RequestInteraction](https://www.weex.com/api-doc/ai/QuickStart/RequestInteraction)
2. The platform provides two types of APIs:
	- **Public APIs:** Used to access configuration details, market data, and other public information. No authentication is required.
	- **Private APIs:** Used for order management, account operations, and other sensitive actions. Authentication is mandatory.

When calling private APIs, please include the following authentication details in your HTTP request headers:

| Request Header Fields | Information |
| --- | --- |
| ACCESS-KEY | A unique identifier for the user account |
| ACCESS-PASSPHRASE | The password associated with the API Key |
| ACCESS-TIMESTAMP | A Unix Epoch timestamp in milliseconds. The timestamp is valid for 30 seconds and must match the one used in the signature calculation. |
| ACCESS-SIGN | The request signature string. You may use the signature generation method provided in the sample code. For the underlying algorithm, refer to the signature documentation: [https://www.weex.com/api-doc/ai/QuickStart/Signature](https://www.weex.com/api-doc/ai/QuickStart/Signature) |
| Content-Type | Content fixed as application/json. |
| locale | Language identifier (e.g.,zh-CN,en-US) |

Fill in " [https://api-contract.weex.com](https://api-contract.weex.com/) " for BASE\_URL

HTTP status codes:

- 200 Success – Successful response
- 400 Bad Request – Invalid request format
- 403 Forbidden – You do not have access to the requested resource
- 404 Not Found – Request not found
- 429 Too Many Requests – Request too frequent, please try again later
- 500 Internal Server Error – We had a problem with our server
- 521 Web Server is Down – IP not whitelisted

Other error codes： [https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode)

You can start API testing once the above steps are completed. The following instructions use BTC as the example asset.

#### Check Account Balance

1. Use Account Balance API to request your current account balance

Sample code:

```python
import time
import hmac
import hashlib
import base64
import requests

api_key = ""
secret_key = ""
access_passphrase = ""

def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
  message = timestamp + method.upper() + request_path + query_string
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_get(api_key, secret_key, access_passphrase, method, request_path, query_string):
  timestamp = str(int(time.time() * 1000))
  signature = generate_signature_get(secret_key, timestamp, method, request_path, query_string)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }

  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string, headers=headers)
  return response

def assets():
    request_path = "/capi/v2/account/assets"
    query_string = ""
    response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    assets()
```
1. Check the response

The example below indicates a successful response

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

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

#### Get Asset Price

1.Use Price Ticker API to request the latest price of `cmt_btcusdt`

Sample code:

```python
import requests

def send_request_get( method, request_path, query_string):
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string)
  return response

def ticker():
    request_path = "/capi/v2/market/ticker"
    query_string = "?symbol=cmt_btcusdt"
    response = send_request_get( "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    ticker()
```
1. Check the response

The example below indicates a successful response

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

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

#### Set Leverage

**Note:** The maximum leverage allowed in this competition is 20x; please follow the rules on WEEX official website.

1. Use the [Leverage Adjustment API](https://www.weex.com/api-doc/contract/Account_API/AdjustLeverage) to modify the cross-margin leverage for `cmt_btcusdt`

Sample code:

```python
import time
import hmac
import hashlib
import base64
import requests
import json

api_key = ""
secret_key = ""
access_passphrase = ""

def generate_signature(secret_key, timestamp, method, request_path, query_string, body):
  message = timestamp + method.upper() + request_path + query_string + str(body)
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_post(api_key, secret_key, access_passphrase, method, request_path, query_string, body):
  timestamp = str(int(time.time() * 1000))
  body = json.dumps(body)
  signature = generate_signature(secret_key, timestamp, method, request_path, query_string, body)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "POST":
    response = requests.post(url + request_path, headers=headers, data=body)
  return response

def leverage():
    request_path = "/capi/v2/account/leverage"
    body = {"symbol":"cmt_btcusdt","marginMode":1,"longLeverage":"1","shortLeverage":"1"}
    query_string = ""
    response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    leverage()
```

You may set other leverage values (up to 20×) in the same way—simply replace the leverage numbers accordingly. **The leverage values are followed by the trading pairs.**

2.Check the response

The example below indicates a successful response

```json
{
    "msg": "success",
    "requestTime": 1713339011237,
    "code": "200"
}
```

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

#### Place order

**Note:** Only the following trading pairs are allowed in this competition:`cmt_btcusdt`, `cmt_ethusdt`, `cmt_solusdt`, `cmt_dogeusdt`, `cmt_xrpusdt`, `cmt_adausdt`, `cmt_bnbusdt`, `cmt_ltcusdt`

The maximum leverage is **20x**; please follow the rules on WEEX official website.

1.Use [Get Futures Information](https://www.weex.com/api-doc/contract/Market_API/GetContractInfo) to retrieve contract information for `cmt_btcusdt` (order precision, price precision, max/min order size, etc.)

Sample code:

```python
import requests

def send_request_get( method, request_path, query_string):
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string)
  return response

def contracts():
    request_path = "/capi/v2/market/contracts"
    query_string = "?symbol=cmt_btcusdt"
    response = send_request_get( "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    contracts()
```

2.Check the response

The example below indicates a successful response

```json
[
  {
    "symbol": "cmt_btcusdt",
    "underlying_index": "BTC",
    "quote_currency": "USDT",
    "coin": "USDT",
    "contract_val": "0.0001",
    "delivery": [
      "00:00:00",
      "08:00:00",
      "16:00:00"
    ],
    "size_increment": "4",
    "tick_size": "1",
    "forwardContractFlag": true,
    "priceEndStep": 1,
    "minLeverage": 1,
    "maxLeverage": 400,
    "buyLimitPriceRatio": "0.01",
    "sellLimitPriceRatio": "0.01",
    "makerFeeRate": "0.0002",
    "takerFeeRate": "0.0008",
    "minOrderSize": "0.0001",
    "maxOrderSize": "1200",
    "maxPositionSize": "1000000",
    "marketOpenLimitSize": "100"
  }
]
```

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

**Note:** The API response may show a higher maximum leverage (for example, 400x) available on the platform. However, for this competition, the maximum leverage you are allowed to use is strictly limited to 20x.

1. Use [Place Order API](https://www.weex.com/api-doc/contract/Transaction_API/PlaceOrder) to open a long position for `cmt_btcusdt` with a limit price of **100000.0** and **0.0001 BTC**, using the contract information from Step 1 (order precision, price precision, and max/min order size) to construct the parameters.

Sample code:

```python
import time
import hmac
import hashlib
import base64
import requests
import json

api_key = ""
secret_key = ""
access_passphrase = ""

def generate_signature(secret_key, timestamp, method, request_path, query_string, body):
  message = timestamp + method.upper() + request_path + query_string + str(body)
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_post(api_key, secret_key, access_passphrase, method, request_path, query_string, body):
  timestamp = str(int(time.time() * 1000))
  body = json.dumps(body)
  signature = generate_signature(secret_key, timestamp, method, request_path, query_string, body)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }
  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "POST":
    response = requests.post(url + request_path, headers=headers, data=body)
  return response

def placeOrder():
    request_path = "/capi/v2/order/placeOrder"
    body = {
        "symbol": "cmt_btcusdt",
        "client_oid": "test",
        "size": "0.0001",
        "type": "1",
        "order_type": "0",
        "match_price": "0",
        "price": "100000.0"}
    query_string = ""
    response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    placeOrder()
```
1. Check the response

The example below indicates a successful response

```json
{
        "client_oid": null,
        "order_id": "596471064624628269"
}
```

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

#### Get Trade Details for Completed Orders

1.Use Trade Details API to retrieve your trade history.

Sample code:

```python
import time
import hmac
import hashlib
import base64
import requests

api_key = ""
secret_key = ""
access_passphrase = ""

def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
  message = timestamp + method.upper() + request_path + query_string
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  return base64.b64encode(signature).decode()

def send_request_get(api_key, secret_key, access_passphrase, method, request_path, query_string):
  timestamp = str(int(time.time() * 1000))
  signature = generate_signature_get(secret_key, timestamp, method, request_path, query_string)
  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }

  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string, headers=headers)
  return response

def fills():
    request_path = "/capi/v2/order/fills"
    query_string = "?symbol=cmt_btcusdt&orderId=YOUR_ORDER_ID"
    response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)
if __name__ == '__main__':
    fills()
```

2.Check the response

The example below indicates a successful response

If the request fails, refer to the detailed error codes [here](https://www.weex.com/api-doc/contract/ErrorCodes/ExampleOfErrorCode), or contact technical support in the [TG group](https://t.me/weexaiwars).

**Our customer support team will contact you once you pass the testing.**

### 2.3 Funding & Model Testing

Once you have completed API testing and passed the qualification review, your account will receive the initial funds required for AI model testing. You may freely use these funds until January 5, 2026 to optimize your AI model.

If you encounter any technical issues or run out of test funds, please contact our official technical support group for assistance.

### 2.4 Pre-Competition Preparation & Account Reset

After the model testing phase, the official list of participants will be published, so please stay updated via the [official participant list](https://dorahacks.io/hackathon/weex-ai-trading/faq) on the event page. To ensure fairness before the official competition begins, all participant accounts will be reset to a unified initial state, with your competition fund balances reset to **1,000 USDT**, all open orders canceled, and all positions closed.

At this point, all pre-competition preparations are complete. Ensure your AI model is fully integrated with your API Key and ready to trade immediately once the competition begins.

**Note:** All official announcements, participant lists, and rule updates will be published on the [WEEX official event page](https://www.weex.com/events/ai-trading). Detailed schedules, ranking rules, and risk management terms will be provided in the official Competition Rules Handbook or via separate notices before the event.

### Reference

- [FAQ](https://dorahacks.io/hackathon/weex-ai-trading/faq)
- For specific inquiries or additional support: [Ask Question](https://dorahacks.io/hackathon/weex-ai-trading/qa)
- **Hackathon Timeline** Pre-Registration: Now – December 30, 2025 Pre-Season (Online): Early January 2026 (20 days) Finals (Online): Late February 2026 (17 days) Awarding Ceremony (Dubai): March 2026

You now have all the information needed to successfully register, prepare, and participate in **AI Wars: WEEX Alpha Awakens**. Follow each step carefully to ensure your AI model is fully integrated, tested, and ready for competition day.

Register now to secure your spot: [https://www.weex.com/events/ai-trading](https://www.weex.com/events/ai-trading). Good luck, and may the best algorithms win!

API keys have rate limits. Exceeding them returns error 429: Too many requests. The account is used as the basic unit of speed limit for the endpoints that need to carry an API key. For endpoints that do not need to carry an API key, IP addresses are used as the basic unit of rate limiting.

**Limits description**

IP-based and UID-based limits operate independently.

Each endpoint indicates whether it follows IP or UID limits, along with its weight value.

Endpoints with IP limits have an independent limit of 1000 requests per 10 seconds.

Endpoints with UID limits also have an independent limit of 1000 requests per 10 seconds.

**Special Interface Description**

The interfaces for Place Order, Cancel Order, Place Trigger Order, and Cancel Trigger Order are limited to a maximum of 10 requests per second.

**Limits error**

When a 429 error occurs, make sure to stop sending excessive requests.

## API Domain

You can use different domain as below Rest API.

| Domain Name | API | Description |
| --- | --- | --- |
| REST Domain | [https://api-contract.weex.com](https://api-contract.weex.com/) | Main Domain |

The ACCESS-SIGN request header is generated by using the **HMAC SHA256** method encryption on the **timestamp + method.toUpperCase() + requestPath + "?" + queryString + body** string (+ denotes string concatenation), and putting the result through **BASE64** encoding.

**Signature Field Description**

- timestamp: This matches the ACCESS-TIMESTAMP header.
- method: The request method (POST/GET), with all letters in uppercase.
- requestPath: API endpoint path.
- queryString: The query parameters after the "?" in the URL.
- body: The string that corresponds to the request body (omitted if empty, typically for GET requests).

**Signature format rules if queryString is empty**

- timestamp + method.toUpperCase() + requestPath + body

**Signature format rules if queryString is not empty**

- timestamp + method.toUpperCase() + requestPath + "?" + queryString + body

**Examples**

Fetching market depth, using BTCUSDT as an example:

- Timestamp = 1591089508404
- Method = "GET"
- requestPath = "/capi/v2/market/candles"
- queryString= "?symbol=cmt\_btcusdt&limit=20"

**Generate the string to be signed:**

- '1591089508404GET/api/v2/market/depth?symbol=cmt\_btcusdt&limit=20'

Placing an order, using cmt\_btcusdt as an example:

- Timestamp = 1561022985382
- Method = "POST"
- requestPath = "/capi/v2/order/placeOrder"
- body =
	```json
	{"symbol":"cmt_btcusdt","quantity":"8","side":"buy","price":"1","orderType":"limit","clientOrderId":"ww#123456"}
	```

**Generate the string to be signed:**

- ```markdown
	'1561022985382POST/capi/v2/order/placeOrder{"symbol":"cmt_btcusdt","size":"8","side":"buy","price":"1","orderType":"limit","clientOrderId":"ww#123456"}'
	```

**Steps to generate the final signature**

1. Encrypt the unsigned string with HMAC SHA256 using your secretKey
	- Signature = hmac\_sha256(secretkey, Message)
2. Encode the signature using Base64
	- Signature = base64.encode(Signature)

    - Java
- Python

```python
import time
import hmac
import hashlib
import base64
import requests
import json

api_key = ""
secret_key = ""
access_passphrase = ""

def generate_signature(secret_key, timestamp, method, request_path, query_string, body):
  message = timestamp + method.upper() + request_path + query_string + str(body)
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  # print(base64.b64encode(signature).decode())
  return base64.b64encode(signature).decode()

def generate_signature_get(secret_key, timestamp, method, request_path, query_string):
  message = timestamp + method.upper() + request_path + query_string
  signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
  # print(base64.b64encode(signature).decode())
  return base64.b64encode(signature).decode()

def send_request_post(api_key, secret_key, access_passphrase, method, request_path, query_string, body):
  timestamp = str(int(time.time() * 1000))
  # print(timestamp)
  body = json.dumps(body)
  signature = generate_signature(secret_key, timestamp, method, request_path, query_string, body)

  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }

  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path, headers=headers)
  elif method == "POST":
    response = requests.post(url + request_path, headers=headers, data=body)
  return response

def send_request_get(api_key, secret_key, access_passphrase, method, request_path, query_string):
  timestamp = str(int(time.time() * 1000))
  # print(timestamp)
  signature = generate_signature_get(secret_key, timestamp, method, request_path, query_string)

  headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": access_passphrase,
        "Content-Type": "application/json",
        "locale": "en-US"
  }

  url = "https://api-contract.weex.com/"  # Please replace with the actual API address
  if method == "GET":
    response = requests.get(url + request_path+query_string, headers=headers)
  return response

def get():
    # Example of calling a GET request
    request_path = "/capi/v2/account/position/singlePosition"
    query_string = '?symbol=cmt_btcusdt'
    response = send_request_get(api_key, secret_key, access_passphrase, "GET", request_path, query_string)
    print(response.status_code)
    print(response.text)

def post():
    # Example of calling a POST request
    request_path = "/capi/v2/order/placeOrder"
    body = {
    "symbol": "cmt_btcusdt",
    "client_oid": "71557515757447",
    "size": "0.01",
    "type": "1",
    "order_type": "0",
    "match_price": "1",
    "price": "80000"}
    query_string = ""
    response = send_request_post(api_key, secret_key, access_passphrase, "POST", request_path, query_string, body)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    get()
    post()
```

All requests are based on the HTTPS protocol. The Content-Type in the request headers must be set to 'application/json'.

**Request Processing**

- Request parameters: Parameter encapsulation according to endpoint request parameter specification.
- Submit request: Submit the encapsulated parameters to the server via GET/POST.
- Server response: The server first performs security checks on the request data, and after passing the check, returns the response data to the user in the JSON format based on the operation logic.
- Data processing: Process the server response data.

**Success**

HTTP 200 status codes indicates success and may contain content.Response content (if any) will be included in the returned data.

**Common error codes**

- 400 Bad Request – Invalid request format
- 401 Unauthorized – Invalid API Key
- 403 Forbidden – You do not have access to the requested resource
- 404 Not Found — No requests found
- 429 Too Many Requests – Rate limit exceeded
- 500 Internal Server Error – We had a problem with our server
- Failed responses include error descriptions in the body.

**Timestamp**

The ACCESS-TIMESTAMP in request signatures is in milliseconds.Requests are rejected if the timestamp deviates by over 30 seconds from the API server time. If the local server time deviates significantly from the API server time, we recommend updating the HTTP header by querying the API server time.

**Request formats**

Only two request methods are supported: GET and POST

- GET: Parameters are sent via queryString in the path to the server.
- POST: Parameters are sent as a JSON-formatted body to the server.

- **POST** `/capi/v2/order/uploadAiLog`

Weight(IP): 1, Weight(UID): 1

## Important rule

BUIDLs entering the live trading phase must provide an AI log (ai\_log) containing:

- Model version
- Input and output data
- Order execution details

The AI log is required to verify AI involvement and compliance. If you fail to provide valid proof of AI involvement, we will disqualify your team and remove it from the rankings. Only approved UIDs on the official allowlist may submit AI log data.

**Request parameters**

| Parameter | Type | Required? | Description |
| --- | --- | --- | --- |
| orderId | Long | No | Order ID |
| stage | String | Yes | Stage identifier |
| model | String | Yes | Model name |
| input | JSON | Yes | Input parameters |
| output | JSON | Yes | Output results |
| explanation | String | Yes | A concise, explanatory summary of AI's behavior. Used to describe the AI’s analysis, reasoning, or output in natural language. The content should not exceed 500 words. |

  
  

**Request example 1**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/uploadAiLog" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{
    "orderId": null,
    "stage": "Decision Making",
    "model": "GPT-5-mini",
    "input": {"prompt":"Summarize last 6h BTC/ETH correlation and give a directional signal."},
    "output": {"response":"Sell ETH; correlation weakened, BTC showing dominance."},
    "explanation": "Analysis of the past 6 hours of market data indicates a weakening correlation between BTC and ETH. BTC demonstrated relative strength and capital dominance, resulting in a directional signal favoring selling ETH."
}
'
```

**Request example 2**

```powershell
curl -X POST "https://api-contract.weex.com/capi/v2/order/uploadAiLog" \
   -H "ACCESS-KEY:*******" \
   -H "ACCESS-SIGN:*" \
   -H "ACCESS-PASSPHRASE:*" \
   -H "ACCESS-TIMESTAMP:1659076670000" \
   -H "locale:zh-CN" \
   -H "Content-Type: application/json" \
   -d '{
    "orderId": null,
    "stage": "Strategy Generation",
    "model": "GPT-5-turbo",
    "input": {
        "prompt": "Predict BTC/USDT price trend for the next 3 hours.",
        "data": {
            "RSI_14": 36.8,
            "EMA_20": 68950.4,
            "FundingRate": -0.0021,
            "OpenInterest": 512.3
        }
    },
    "output": {
        "signal": "Buy",
        "confidence": 0.82,
        "target_price": 69300,
        "reason": "Negative funding + rising open interest implies short squeeze potential."
    },
     "explanation": "Low RSI and price near the EMA20 suggest weakening downside momentum. Negative funding with rising open interest points to short-side pressure and potential squeeze risk, indicating a bullish bias for BTC over the next three hours."
}'
```

**Response parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| code | String | Request status code, "00000" indicates success |
| msg | String | Request result description, "success" indicates success |
| requestTime | Long | Request timestamp (milliseconds) |
| data | String | Returned business data, "upload success" indicates upload successful |

**Response example**

```json
{
  "code": "00000",
  "msg": "success",
  "requestTime": 1763103201300,
  "data": "upload success"
}
```