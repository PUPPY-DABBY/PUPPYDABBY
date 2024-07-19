from binance.um_futures import UMFutures
import ta
from ta.trend import MACD
import pandas as pd
from time import sleep
from binance.error import ClientError
import requests

api = 'jQXVa7PXKVr18687yodohcmU2z1zTntdtPf5FuFtBYUcrtWUbDc5nLSjFePCR6RM' 
secret = 'zEgWcFbahAGbMzVIHWrx3O7VGjCVWrrCXzGGSO6jfrFbVp9oLjzRxVaA3eszHPW6'

client = UMFutures(key = api, secret=secret)

def get_public_ip():
    try:
        # Use an IP lookup service to get your public IP
        response = requests.get('https://api64.ipify.org?format=json')
        response.raise_for_status()  # Raise an exception for bad responses
        public_ip = response.json()['ip']
        return public_ip
    
    except ClientError as error:
         print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

# Get and print the public IP address
public_ip = get_public_ip()
if public_ip:
    print(f'Your public IP address: {public_ip}')
else:
    print('Failed to retrieve the public IP address.')


tp = 0.04
sl = 0.02
volume = 10  
leverage = 25
type = 'CROSS'
qty = 100  

# getting your futures balance in USDT
def get_balance_usdt():
    try:
        response = client.balance(recvWindow=6000)
        for elem in response:
            if elem['asset'] == 'USDT':
                return float(elem['balance'])

    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Getting all available symbols on the Futures ('BTCUSDT', 'ETHUSDT', ....)
def get_tickers_usdt():
    tickers = []
    resp = client.ticker_price()
    for elem in resp:
        if 'USDT' in elem['symbol']:
            tickers.append(elem['symbol'])
    return tickers


# Getting candles for the needed symbol, its a dataframe with 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'
def klines(symbol):
    try:
        resp = pd.DataFrame(client.klines(symbol, '1h'))
        resp = resp.iloc[:,:6]
        resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        resp = resp.set_index('Time')
        resp.index = pd.to_datetime(resp.index, unit = 'ms')
        resp = resp.astype(float)
        return resp
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Set leverage for the needed symbol. You need this bcz different symbols can have different leverage
def set_leverage(symbol, level):
    try:
        response = client.change_leverage(
            symbol=symbol, leverage=level, recvWindow=6000
        )
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# The same for the margin type
def set_mode(symbol, type):
    try:
        response = client.change_margin_type(
            symbol=symbol, marginType=type, recvWindow=6000
        )
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Price precision. BTC has 1, XRP has 4
def get_price_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['pricePrecision']


# Amount precision. BTC has 3, XRP has 1
def get_qty_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['quantityPrecision']


# Open new order with the last price, and set TP and SL:
def open_order(symbol, side):
    price = float(client.ticker_price(symbol)['price'])
    qty_precision = get_qty_precision(symbol)
    price_precision = get_price_precision(symbol)
    qty = round(volume/price, qty_precision)
    if side == 'buy':
        try:
            resp1 = client.new_order(symbol=symbol, side='BUY', type='LIMIT', quantity=qty, timeInForce='GTC', price=price)
            print(symbol, side, "placing order")
            print(resp1)
            sleep(2)
            sl_price = round(price - price*sl, price_precision)
            resp2 = client.new_order(symbol=symbol, side='SELL', type='STOP_MARKET', quantity=qty, timeInForce='GTC', stopPrice=sl_price)
            print(resp2)
            sleep(2)
            tp_price = round(price + price * tp, price_precision)
            resp3 = client.new_order(symbol=symbol, side='SELL', type='TAKE_PROFIT_MARKET', quantity=qty, timeInForce='GTC',
                                     stopPrice=tp_price)
            print(resp3)
        except ClientError as error:
            print(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
    if side == 'sell':
        try:
            resp1 = client.new_order(symbol=symbol, side='SELL', type='LIMIT', quantity=qty, timeInForce='GTC', price=price)
            print(symbol, side, "placing order")
            print(resp1)
            sleep(2)
            sl_price = round(price + price*sl, price_precision)
            resp2 = client.new_order(symbol=symbol, side='BUY', type='STOP_MARKET', quantity=qty, timeInForce='GTC', stopPrice=sl_price)
            print(resp2)
            sleep(2)
            tp_price = round(price - price * tp, price_precision)
            resp3 = client.new_order(symbol=symbol, side='BUY', type='TAKE_PROFIT_MARKET', quantity=qty, timeInForce='GTC',
                                     stopPrice=tp_price)
            print(resp3)
        except ClientError as error:
            print(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

# Your current positions (returns the symbols list):
def get_pos():
    try:
        resp = client.get_position_risk()
        pos = []
        for elem in resp:
            if float(elem['positionAmt']) != 0:
                pos.append(elem['symbol'])
        return pos
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

def check_orders():
    try:
        response = client.get_orders(recvWindow=6000)
        sym = []
        for elem in response:
            sym.append(elem['symbol'])
        return sym
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

# Close open orders for the needed symbol. If one stop order is executed and another one is still there
def close_open_orders(symbol):
    try:
        response = client.cancel_open_orders(symbol=symbol, recvWindow=6000)
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Strategy. Can use any other:
def str_signal(symbol):
    kl = klines(symbol)
    rsi = ta.momentum.RSIIndicator(kl.Close).rsi()
    rsi_k = ta.momentum.StochRSIIndicator(kl.Close).stochrsi_k()
    rsi_d = ta.momentum.StochRSIIndicator(kl.Close).stochrsi_d()
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    
    if rsi.iloc[-1] < 40 and ema.iloc[-1] < kl.Close.iloc[-1]:
        return 'up'
    elif rsi.iloc[-1] > 60 and ema.iloc[-1] > kl.Close.iloc[-1]:
        return 'down'
    else:
        return 'none'

def rsi_signal(symbol):
    kl = klines(symbol)
    rsi = ta.momentum.RSIIndicator(kl.Close).rsi()
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    if rsi.iloc[-2] < 30 and rsi.iloc[-1] > 30:
        return 'up'
    if rsi.iloc[-2] > 70 and rsi.iloc[-1] < 70:
        return 'down'

    else:
        return 'none'
    
def calculate_sma(close_prices, window=20):
    return ta.trend.sma_indicator(close_prices, window=window)

def calculate_ema(close_prices, window=200):
    return ta.trend.ema_indicator(close_prices, window=window)

def calculate_rsi(close_prices, window=8):
    return ta.momentum.rsi(close_prices, window=window)

def calculate_macd(close_prices, window_fast=12, window_slow=26):
    return ta.trend.macd(close_prices, window_fast=window_fast, window_slow=window_slow)

def calculate_bb_upper(close_prices, window=20, window_dev=2):
    return ta.volatility.bollinger_hband(close_prices, window=window, window_dev=window_dev)

def calculate_bb_lower(close_prices, window=20, window_dev=2):
    return ta.volatility.bollinger_lband(close_prices, window=window, window_dev=window_dev)

def calculate_stoch_rsi(close_prices, window=14, smooth1=3):
    stoch_rsi = ta.momentum.StochRSIIndicator(close_prices, window=window, smooth1=smooth1)
    stoch_k = stoch_rsi.stochrsi_k()
    stoch_d = stoch_rsi.stochrsi_d()
    return stoch_k, stoch_d

def generate_signal(macd, sma, ema, bb_upper, close_prices, stoch_k, stoch_d, rsi):
    if (macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and
        sma.iloc[-1] > ema.iloc[-1] and bb_upper.iloc[-1] > close_prices.iloc[-1]
        and stoch_k.iloc[-1] > 30 and stoch_d.iloc[-1] > 30 and rsi.iloc[-1] > 30):
        return 'up'
    if (macd.iloc[-3] < 0 and macd.iloc[-2] < 0 and macd.iloc[-1] > 0 and
        sma.iloc[-1] < ema.iloc[-1] and bb_upper.iloc[-1] < close_prices.iloc[-1]
        and stoch_k.iloc[-1] < 70 and stoch_d.iloc[-1] < 70 and rsi.iloc[-1] < 70):
        return 'down'
    else:
        return 'none'


def test_strategy(symbol):
    kl = klines(symbol)
    close_prices = kl['Close']
    
    sma = calculate_sma(close_prices)
    ema = calculate_ema(close_prices)
    rsi = calculate_rsi(close_prices)
    macd = calculate_macd(close_prices)
    bb_upper = calculate_bb_upper(close_prices)
    bb_lower = calculate_bb_lower(close_prices)
    stoch_k, stoch_d = calculate_stoch_rsi(close_prices)
    
    return generate_signal(macd, sma, ema, bb_upper, close_prices, stoch_k, stoch_d, rsi)
       


def combined_strategy1(symbol):
    kl = klines(symbol)
    macd = ta.trend.macd_diff(kl.Close)
    stoch_rsi = ta.momentum.StochRSIIndicator(kl['Close'], window=14, smooth1=3)
    stoch_k = stoch_rsi.stochrsi_k()
    stoch_d = stoch_rsi.stochrsi_d()
    ema = ta.trend.ema_indicator(kl.Close, window=200)

    if (macd.iloc[-3] < 0 and macd.iloc[-2] < 0 and macd.iloc[-1] > 0 and
        ema.iloc[-1] < kl.Close.iloc[-1] and
        stoch_k.iloc[-1] < 30 and stoch_d.iloc[-1] < 30):
        return 'up'
    elif (macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and
          ema.iloc[-1] > kl.Close.iloc[-1] and
          stoch_k.iloc[-1] < 70 and stoch_d.iloc[-1] < 70):
        return'down'
    else:
        return 'none'
    
def calculate_kdj(close, low, high, window=30, k_window=3, d_window=3):
    # Calculate KDJ
    rsv = (close - low.rolling(window=window).min()) / (high.rolling(window=window).max() - low.rolling(window=window).min()) * 100
    kdj_k = rsv.ewm(span=k_window, adjust=False).mean()
    kdj_d = kdj_k.ewm(span=d_window, adjust=False).mean()
    return kdj_k, kdj_d

def combined_strategy(symbol):
    kl = klines(symbol)

    # Calculate indicators
    macd = ta.trend.macd_diff(kl['Close'])
    kdj_k, kdj_d = calculate_kdj(kl['Close'], kl['Low'], kl['High'])

    # Check strategy conditions
    if (
        macd.iloc[-3] < 0
        and macd.iloc[-2] < 0
        and macd.iloc[-1] > 0
        and kdj_k.iloc[-1] < 20
        and kdj_d.iloc[-1] < 20
    ):
      return 'up'
    elif (
        macd.iloc[-3] > 0
        and macd.iloc[-2] > 0
        and macd.iloc[-1] < 0
        and kdj_k.iloc[-1] > 80
        and kdj_d.iloc[-1] > 80
    ):
      return 'down'
    else:
        return 'none'

def macd_ema(symbol):
    print("Running macd_ema...")
    kl = klines(symbol)
    macd = ta.trend.macd_diff(kl.Close)
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    if macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and ema.iloc[-1] > kl.Close.iloc[-1]: 
        return 'up'
    elif macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and ema.iloc[-1] > kl.Close.iloc[-1]:
        print("macd_ema signal: down")
        return 'down'
    else:
        print("macd_ema signal: none")
        return 'none'

def kj_strategy(symbol, cci_period=20, atr_multiplier=1, atr_period=5):
    kl = klines(symbol)
    atr = ta.volatility.average_true_range(high=kl['High'], low=kl['Low'], close=kl['Close'], window=atr_period)
    cci = ta.trend.cci(kl['High'], kl['Low'], kl['Close'], window=cci_period)
    up_threshold = kl['Low'] - atr * atr_multiplier 
    down_threshold = kl['High'] + atr * atr_multiplier
    magic_trend = [0.0] * len(kl)

    for i in range(1, len(kl)):
        if cci.iloc[i] >= 0:
            magic_trend[i] = up_threshold.iloc[i] if up_threshold.iloc[i] < magic_trend[i - 1] else magic_trend[i - 1]
        else:
            magic_trend[i] = down_threshold.iloc[i] if down_threshold.iloc[i] > magic_trend[i - 1] else magic_trend[i - 1]

        # Generate signals based on crossovers and crossunders
        if kl['Close'].iloc[i] > magic_trend[i] and kl['Close'].iloc[i - 1] <= magic_trend[i - 1]:
            return 'up'
        elif kl['Close'].iloc[i] < magic_trend[i] and kl['Close'].iloc[i - 1] >= magic_trend[i - 1]:
            return 'down'
        else:
            return 'none'

def kj(symbol):
    macd = macd_ema(symbol)
    kjs = kj_strategy(symbol)

    if kjs == 'up' and macd == 'up':
        return 'up'
    elif  kjs == 'down' and macd == 'down':
        return 'down'
    else:
        return 'none'


def kj_strategy1(symbol, cci_period=20, atr_multiplier=1, atr_period=5):
    kl = klines(symbol)
    
    # Calculate MACD
    macd = ta.trend.macd_diff(kl['Close'])
    
    # Calculate EMAs
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    
    atr = ta.volatility.average_true_range(high=kl['High'], low=kl['Low'], close=kl['Close'], window=atr_period)
    cci = ta.trend.cci(kl['High'], kl['Low'], kl['Close'], window=cci_period)
    up_threshold = kl['Low'] - atr * atr_multiplier
    down_threshold = kl['High'] + atr * atr_multiplier
    magic_trend = [0.0] * len(kl)

    for i in range(1, len(kl)):
        if cci.iloc[i] >= 0:
            magic_trend[i] = up_threshold.iloc[i] if up_threshold.iloc[i] < magic_trend[i - 1] else magic_trend[i - 1]
        else:
            magic_trend[i] = down_threshold.iloc[i] if down_threshold.iloc[i] > magic_trend[i - 1] else magic_trend[i - 1]

        # Generate signals based on crossovers and crossunders
        if kl['Close'].iloc[i] > magic_trend[i] and kl['Close'].iloc[i - 1] <= magic_trend[i - 1] and macd.iloc[i] > 0:
            return 'up'
        elif kl['Close'].iloc[i] < magic_trend[i] and kl['Close'].iloc[i - 1] >= magic_trend[i - 1] and macd.iloc[i] < 0:
            return 'down'
        else:
            return 'none'


def sma(symbol):
    kl = klines(symbol)
    sma_signals = []

    for window in range(1, 101):
        sma = ta.trend.sma_indicator(kl.Close, window=window)
        sma_signals.append(sma.iloc[-1])

    if all(sma_signals[i] > sma_signals[i + 1] for i in range(len(sma_signals) - 1)) and kl.Close.iloc[-1] > sma_signals[-1]:
        return 'up'
    elif all(sma_signals[i] < sma_signals[i + 1] for i in range(len(sma_signals) - 1)) and kl.Close.iloc[-1] < sma_signals[-1]:
        return 'down'
    else:
        return 'none'
        
def combination(symbol):
    kj = kj_strategy(symbol)
    sma_signal = sma(symbol)

    if kj == 'up' and sma_signal == 'up':
        return 'up'
    elif kj == 'down' and sma_signal == 'down':
        return 'down'
    else:
        return 'none'

def new_strategy(symbol):
    kl = klines(symbol)
    print(get_tickers_usdt)
    macd = ta.trend.macd_diff(kl['Close']) 
    print("Running1...")
    rsi = ta.momentum.RSIIndicator(kl['Close']).rsi()
    print("Running2...")
    obv = ta.volume.OnBalanceVolumeIndicator(close=kl['Close'], volume=kl['Volume']).on_balance_volume()
    aroon_indicator = ta.trend.AroonIndicator(high=kl['High'], low=kl['Low'], window=25)
    aroon_oscillator = aroon_indicator.aroon_up() - aroon_indicator.aroon_down()
    
    # Calculate Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(high=kl['High'], low=kl['Low'], close=kl['Close'])
    K = stochastic.stoch()
    D = stochastic.stoch_signal()
    print("Running5...")

    print("MACD:", macd.iloc[-3:])
    print("RSI:", rsi.iloc[-1])
    print("Aroon Oscillator:", aroon_oscillator.iloc[-1])
    print("OBV:", obv.iloc[-1])
    print("Stochastic Oscillator (K, D):", K.iloc[-1], D.iloc[-1])


    if (macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and
        rsi.iloc[-1] > 50 and aroon_oscillator.iloc[-1] < 0 and obv.iloc[-1] > obv.iloc[-2] and
        K.iloc[-1] < D.iloc[-1] and K.iloc[-2] > D.iloc[-2] and K.iloc[-1] > 50):
        print("Running00000...")
        return 'down'
    if (macd.iloc[-3] < 0 and macd.iloc[-2] < 0 and macd.iloc[-1] > 0 and
        rsi.iloc[-1] < 30 and aroon_oscillator.iloc[-1] > 0 and obv.iloc[-1] < obv.iloc[-2] and
        K.iloc[-1] > D.iloc[-1] and K.iloc[-2] < D.iloc[-2] and K.iloc[-1] < 30):
        print("Running10...")
        return 'up'
    else:
        return 'none'

orders = 0
symbol = ''
# getting all symbols from Binace Futures list:
symbols = get_tickers_usdt()

while True:
    # we need to get balance to check if the connection is good, or you have all the needed permissions
    balance = get_balance_usdt()
    sleep(1)
    if balance == None:
        print('Cant connect to API. Check IP, restrictions or wait some time')
    if balance != None:
        print("My balance is: ", balance, " USDT")
        # getting position list:
        pos = []
        pos1 = get_pos()
        print(f'You have {len(pos)} opened positions:\n{pos}')
        # Getting order list
        ord = []
        ord = check_orders()
        # removing stop orders for closed positions
        for elem in ord:
            if not elem in pos:
                close_open_orders(elem)

        if len(pos) < qty:
            for elem in symbols:
                # Strategies (you can make your own with the TA library):
                #signal = new_strategy(elem)
                signal = kj_strategy(elem)
                #signal = macd_ema(elem)
                #signal = combined_strategy(elem)
                #signal = combined_strategy1(elem)
                #signal = combination(elem)
                #signal = generate_signal(elem)
                #signal = kj(elem)

                # 'up' or 'down' signal, we place orders for symbols that arent in the opened positions and orders
                # we also dont need USDTUSDC because its 1:1 (dont need to spend money for the commission)
                if signal == 'up' and elem != 'USDCUSDT' and not elem in pos and not elem in ord and elem != symbol:
                    print('Found BUY signal for ', elem)
                    set_mode(elem, type)
                    sleep(1)
                    set_leverage(elem, leverage)
                    sleep(1)
                    print('Placing order for ', elem)
                    open_order(elem, 'buy')
                    symbol = elem
                    order = True
                    pos = get_pos()
                    sleep(1)
                    ord = check_orders()
                    sleep(1)
                    sleep(10)
                    # break
                if signal == 'down' and elem != 'USDCUSDT' and not elem in pos and not elem in ord and elem != symbol:
                    print('Found SELL signal for ', elem)
                    set_mode(elem, type)
                    sleep(1)
                    set_leverage(elem, leverage)
                    sleep(1)
                    print('Placing order for ', elem)
                    open_order(elem, 'sell')
                    symbol = elem
                    order = True
                    pos = get_pos()
                    sleep(1)
                    ord = check_orders()
                    sleep(1)
                    sleep(10)
                    # break
    print('Waiting 2 min')
    sleep(120)
