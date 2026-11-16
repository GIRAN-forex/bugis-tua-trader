from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

PAIRS = {
    "FOREX": [
        {"name": "XAU/USD (GOLD)", "tv": "OANDA:XAUUSD", "yf": "GC=F"},
        {"name": "EUR/USD", "tv": "FX:EURUSD", "yf": "EURUSD=X"},
        {"name": "GBP/USD", "tv": "FX:GBPUSD", "yf": "GBPUSD=X"},
        {"name": "USD/JPY", "tv": "FX:USDJPY", "yf": "USDJPY=X"},
        {"name": "GBP/JPY", "tv": "FX:GBPJPY", "yf": "GBPJPY=X"},
        {"name": "AUD/USD", "tv": "FX:AUDUSD", "yf": "AUDUSD=X"},
        {"name": "USD/CAD", "tv": "FX:USDCAD", "yf": "USDCAD=X"},
        {"name": "USD/CHF", "tv": "FX:USDCHF", "yf": "USDCHF=X"},
        {"name": "NZD/USD", "tv": "FX:NZDUSD", "yf": "NZDUSD=X"},
        {"name": "EUR/JPY", "tv": "FX:EURJPY", "yf": "EURJPY=X"},
    ],
    "CRYPTO": [
        {"name": "BTC/USDT", "tv": "BINANCE:BTCUSDT", "yf": "BTC-USD"},
        {"name": "ETH/USDT", "tv": "BINANCE:ETHUSDT", "yf": "ETH-USD"},
        {"name": "SOL/USDT", "tv": "BINANCE:SOLUSDT", "yf": "SOL-USD"},
        {"name": "XRP/USDT", "tv": "BINANCE:XRPUSDT", "yf": "XRP-USD"},
        {"name": "BNB/USDT", "tv": "BINANCE:BNBUSDT", "yf": "BNB-USD"},
        {"name": "DOGE/USDT", "tv": "BINANCE:DOGEUSDT", "yf": "DOGE-USD"},
        {"name": "ADA/USDT", "tv": "BINANCE:ADAUSDT", "yf": "ADA-USD"},
        {"name": "AVAX/USDT", "tv": "BINANCE:AVAXUSDT", "yf": "AVAX-USD"},
    ],
    "NASDAQ": [
        {"name": "US100 (NASDAQ INDEX)", "tv": "NASDAQ:NDX", "yf": "NQ=F"},
        {"name": "NVIDIA (NVDA)", "tv": "NASDAQ:NVDA", "yf": "NVDA"},
        {"name": "APPLE (AAPL)", "tv": "NASDAQ:AAPL", "yf": "AAPL"},
        {"name": "TESLA (TSLA)", "tv": "NASDAQ:TSLA", "yf": "TSLA"},
        {"name": "MICROSOFT (MSFT)", "tv": "NASDAQ:MSFT", "yf": "MSFT"},
        {"name": "AMAZON (AMZN)", "tv": "NASDAQ:AMZN", "yf": "AMZN"},
        {"name": "META (META)", "tv": "NASDAQ:META", "yf": "META"},
        {"name": "GOOGLE (GOOGL)", "tv": "NASDAQ:GOOGL", "yf": "GOOGL"},
    ]
}

TF_MAP = {
    "1": {"yf_interval": "1m", "yf_period": "1d"},
    "5": {"yf_interval": "5m", "yf_period": "5d"},
    "15": {"yf_interval": "15m", "yf_period": "5d"},
    "60": {"yf_interval": "60m", "yf_period": "1mo"},
    "240": {"yf_interval": "60m", "yf_period": "1mo"},
    "D": {"yf_interval": "1d", "yf_period": "6mo"},
}

def analyze_market(yf_ticker, tf_key):
    try:
        tf_info = TF_MAP.get(tf_key, TF_MAP["15"])
        df = yf.download(yf_ticker, period=tf_info["yf_period"], interval=tf_info["yf_interval"], progress=False)
        
        if df.empty or len(df) < 55:
            return {"valid": False, "trend": "BULLISH", "reason": "Menunggu sinkronisasi data pasar real-time..."}

        if tf_key == "240":
            df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Indikator teknikal utama
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR 14
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1])

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        price = float(curr['Close'])
        open_p = float(curr['Open'])
        high_p = float(curr['High'])
        low_p = float(curr['Low'])
        
        prev_open = float(prev['Open'])
        prev_close = float(prev['Close'])
        prev_high = float(prev['High'])
        prev_low = float(prev['Low'])

        rsi = float(curr['RSI']) if not np.isnan(curr['RSI']) else 50.0
        market_trend = "BULLISH" if rsi >= 50 else "BEARISH"

        # Area Resistance & Support 15-20 candle terakhir (mengecualikan candle saat ini)
        resistance_level = float(df['High'].iloc[-20:-1].max())
        support_level = float(df['Low'].iloc[-20:-1].min())

        is_two_dec = any(k in yf_ticker for k in ["GC=F", "JPY", "BTC", "ETH", "SOL", "NQ=F", "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL"])
        decimals = 2 if is_two_dec else 5

        body_size = abs(price - open_p)
        lower_wick = min(open_p, price) - low_p
        upper_wick = high_p - max(open_p, price)

        # ==============================================================
        # 1. DETEKSI BUY (AREA BREAKOUT RESISTANCE ATAU PANTULAN SUPPORT)
        # ==============================================================
        is_bull_breakout = (price > resistance_level) and (price > open_p) and (body_size > (atr * 0.4))
        is_at_support = price <= (support_level + (atr * 1.1))
        is_bull_candle = (price > open_p) and ((lower_wick >= body_size * 0.6) or (price > prev_high))

        if market_trend == "BULLISH" and (is_bull_breakout or (is_at_support and is_bull_candle)):
            entry = round(price, decimals)
            if is_bull_breakout:
                sl = round(resistance_level - (atr * 0.8), decimals)
                tp = round(price + (atr * 2.8), decimals)
                explanation = (
                    f"<b>[SETUP BREAKOUT BUY]</b> Harga berhasil menembus batas Resistance kuat ({round(resistance_level, decimals)}) "
                    f"dengan dorongan candle momentum penuh & RSI Bullish ({round(rsi, 1)}). Siap melanjutkan kenaikan! Gunakan lot 0,01."
                )
            else:
                sl = round(support_level - (atr * 0.6), decimals)
                tp = round(price + (atr * 2.5), decimals)
                explanation = (
                    f"<b>[SETUP DEMAND BOUNCE BUY]</b> Harga memantul di area Support/Demand ({round(support_level, decimals)}) "
                    f"dengan konfirmasi candle Bullish & RSI {round(rsi, 1)}. Patuhi SL dan gunakan lot 0,01."
                )
            return {
                "valid": True, "type": "BUY", "trend": market_trend,
                "signal": "INI VALID BUY (BULLISH)", "entry": entry, "tp": tp, "sl": sl, "explanation": explanation
            }

        # ==============================================================
        # 2. DETEKSI SELL (AREA BREAKOUT SUPPORT ATAU PENOLAKAN RESISTANCE)
        # ==============================================================
        is_bear_breakout = (price < support_level) and (price < open_p) and (body_size > (atr * 0.4))
        is_at_resistance = price >= (resistance_level - (atr * 1.1))
        is_bear_candle = (price < open_p) and ((upper_wick >= body_size * 0.6) or (price < prev_low))

        if market_trend == "BEARISH" and (is_bear_breakout or (is_at_resistance and is_bear_candle)):
            entry = round(price, decimals)
            if is_bear_breakout:
                sl = round(support_level + (atr * 0.8), decimals)
                tp = round(price - (atr * 2.8), decimals)
                explanation = (
                    f"<b>[SETUP BREAKOUT SELL]</b> Harga menjebol batas Support ({round(support_level, decimals)}) ke bawah "
                    f"dengan tekanan jual masif & RSI Bearish ({round(rsi, 1)}). Siap lanjut terjun! Gunakan lot 0,01."
                )
            else:
                sl = round(resistance_level + (atr * 0.6), decimals)
                tp = round(price - (atr * 2.5), decimals)
                explanation = (
                    f"<b>[SETUP SUPPLY REJECTION SELL]</b> Harga ditolak di area Resistance/Supply ({round(resistance_level, decimals)}) "
                    f"dengan konfirmasi candle Bearish & RSI {round(rsi, 1)}. Disiplin gunakan lot 0,01."
                )
            return {
                "valid": True, "type": "SELL", "trend": market_trend,
                "signal": "INI VALID SEL (BEARISH)", "entry": entry, "tp": tp, "sl": sl, "explanation": explanation
            }

        return {
            "valid": False, "trend": market_trend,
            "reason": f"AI memantau Area Breakout & Support/Resistance (Tren: {market_trend} | RSI: {round(rsi, 1)}). Menunggu penembusan atau pantulan terkonfirmasi."
        }

    except Exception as e:
        return {"valid": False, "trend": "BULLISH", "reason": f"Sinkronisasi data: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html', pairs=PAIRS)

@app.route('/auto_scan', methods=['POST'])
def auto_scan():
    data = request.json
    result = analyze_market(data.get('yf_ticker'), data.get('tf'))
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
