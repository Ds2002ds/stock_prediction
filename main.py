from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
import numpy as np
import requests
import time
from datetime import datetime, timezone

# TensorFlow / Keras — only load if .h5 models exist
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available — LSTM models will be skipped")

from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────────────────────
# USER / CONTACT FILES
# ──────────────────────────────────────────────────────────────
# Absolute paths so files are created next to main.py, not wherever gunicorn runs from
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
USER_FILE    = os.path.join(BASE_DIR, "users.json")
CONTACT_FILE = os.path.join(BASE_DIR, "contacts.json")

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(CONTACT_FILE):
    with open(CONTACT_FILE, "w") as f:
        json.dump([], f)

# ──────────────────────────────────────────────────────────────
# LOAD .PKL MODELS
# ──────────────────────────────────────────────────────────────
# Absolute path so gunicorn works from any working directory on Render
MODEL_DIR = os.path.join(BASE_DIR, "models")
models = {}

if os.path.exists(MODEL_DIR):
    print("Loading .pkl models...")
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            name = file.split("_")[0].lower()
            try:
                with open(os.path.join(MODEL_DIR, file), "rb") as f:
                    models[name] = pickle.load(f)
                print(f"  ✓ {name} model loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {file}: {e}")
else:
    print(f"WARNING: Model directory '{MODEL_DIR}' not found")

print("All PKL models:", list(models.keys()))

# ──────────────────────────────────────────────────────────────
# LOAD LSTM MODELS (.h5)
# ──────────────────────────────────────────────────────────────
lstm_models = {}

if TF_AVAILABLE and os.path.exists(MODEL_DIR):
    print("Loading LSTM models...")
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".h5"):
            name = file.split("_")[0].lower()
            try:
                lstm_models[name] = load_model(os.path.join(MODEL_DIR, file))
                print(f"  ✓ LSTM {name} loaded")
            except Exception as e:
                print(f"  ✗ Failed to load LSTM {file}: {e}")

print("All LSTM models:", list(lstm_models.keys()))

# ──────────────────────────────────────────────────────────────
# SCALER & LSTM PREDICTION
# ──────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))

def lstm_predict(df, model):
    try:
        import pandas as pd
        if isinstance(df, list):
            df = pd.DataFrame(df)
            df["Close"] = df["close"]

        close_data = df[["Close"]]
        scaled = scaler.fit_transform(close_data)

        if len(scaled) < 60:
            return None

        last_60 = scaled[-60:]
        X_input = np.array([last_60])
        pred = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred)
        return float(pred_price[0][0])

    except Exception as e:
        print("LSTM error:", e)
        return None

# ──────────────────────────────────────────────────────────────
# YAHOO FINANCE — 3-layer fallback (reliable on Render)
#
# Layer 1: query2 endpoint with rotate User-Agents + crumb
# Layer 2: query1 endpoint  (different server, sometimes less blocked)
# Layer 3: Yahoo v7 quote endpoint (different API path)
#
# Yahoo blocks cloud IPs intermittently. Having 3 independent
# approaches means even if 1-2 are blocked, you get live data.
# ──────────────────────────────────────────────────────────────

import random

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# Persistent session with crumb
_yahoo_session = None
_yahoo_crumb   = None
_last_crumb_time = 0
_CRUMB_TTL = 1800  # refresh crumb every 30 minutes


def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://finance.yahoo.com",
        "Referer": "https://finance.yahoo.com/",
        "DNT": "1",
    })
    return s


def init_yahoo_session(force=False):
    global _yahoo_session, _yahoo_crumb, _last_crumb_time
    now = time.time()

    if not force and _yahoo_session and (now - _last_crumb_time) < _CRUMB_TTL:
        return  # still fresh

    s = _make_session()
    try:
        # Step 1: hit consent/cookie endpoint
        s.get("https://fc.yahoo.com", timeout=5)
        # Step 2: hit finance homepage to set cookies
        s.get("https://finance.yahoo.com", timeout=8)
        # Step 3: get crumb
        for crumb_url in [
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
        ]:
            try:
                r = s.get(crumb_url, timeout=8)
                if r.status_code == 200 and r.text and len(r.text) < 50:
                    _yahoo_crumb = r.text.strip()
                    _last_crumb_time = now
                    print(f"[yahoo] Crumb obtained: {_yahoo_crumb[:8]}...")
                    break
            except Exception:
                continue
        else:
            print("[yahoo] Could not get crumb — will try without it")
    except Exception as e:
        print(f"[yahoo] Session init warning: {e}")

    _yahoo_session = s


def _parse_chart_closes(data):
    """Extract closes list from Yahoo chart JSON response."""
    return [
        c for c in
        data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        if c is not None
    ]


def _parse_chart_records(data):
    """Extract full OHLCV records from Yahoo chart JSON response."""
    chart      = data["chart"]["result"][0]
    timestamps = chart.get("timestamp", [])
    quote      = chart["indicators"]["quote"][0]
    opens      = quote.get("open",   [])
    highs      = quote.get("high",   [])
    lows       = quote.get("low",    [])
    closes     = quote.get("close",  [])
    volumes    = quote.get("volume", [])

    records = []
    for i, ts in enumerate(timestamps):
        try:
            c = closes[i]
            if c is None or c != c:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            records.append({
                "date":   dt,
                "open":   round(float(opens[i]  or c), 2),
                "high":   round(float(highs[i]  or c), 2),
                "low":    round(float(lows[i]   or c), 2),
                "close":  round(float(c),             2),
                "volume": int(volumes[i] or 0),
            })
        except Exception:
            continue
    return records


def _yahoo_chart_request(symbol, interval, yf_range):
    """
    Try fetching Yahoo chart data using 3 independent methods.
    Returns parsed JSON or raises an exception if all fail.
    """
    global _yahoo_session, _yahoo_crumb

    # Ensure session exists
    init_yahoo_session()

    base_params = f"?interval={interval}&range={yf_range}"
    crumb_param = f"&crumb={_yahoo_crumb}" if _yahoo_crumb else ""

    attempts = [
        # Method 1: query2 with crumb (most reliable on cloud)
        {
            "url": f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}{base_params}{crumb_param}",
            "session": _yahoo_session,
        },
        # Method 2: query1 with crumb
        {
            "url": f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{base_params}{crumb_param}",
            "session": _yahoo_session,
        },
        # Method 3: fresh session, no crumb, query2
        {
            "url": f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}{base_params}",
            "session": _make_session(),
        },
    ]

    last_error = None
    for i, attempt in enumerate(attempts):
        try:
            r = attempt["session"].get(attempt["url"], timeout=15)

            # If auth failed, reinit session and retry this specific attempt once
            if r.status_code in (401, 403):
                init_yahoo_session(force=True)
                crumb_param = f"&crumb={_yahoo_crumb}" if _yahoo_crumb else ""
                retry_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}{base_params}{crumb_param}"
                r = _yahoo_session.get(retry_url, timeout=15)

            if r.status_code == 200:
                data = r.json()
                # Validate the response has actual data
                if (data.get("chart", {}).get("result") and
                        data["chart"]["result"][0].get("indicators")):
                    print(f"[yahoo] Method {i+1} succeeded for {symbol}")
                    return data
                else:
                    last_error = f"Empty result from method {i+1}"
                    continue

            last_error = f"HTTP {r.status_code} from method {i+1}"

        except requests.exceptions.Timeout:
            last_error = f"Timeout on method {i+1}"
            print(f"[yahoo] Timeout on method {i+1} for {symbol}")
        except Exception as e:
            last_error = str(e)
            print(f"[yahoo] Method {i+1} error for {symbol}: {e}")

        time.sleep(0.3)  # small pause before next attempt

    raise ValueError(f"All Yahoo Finance methods failed for {symbol}. Last error: {last_error}")


def fetch_yahoo_price(symbol):
    """Fetch current price, change, and % change for a symbol."""
    data   = _yahoo_chart_request(symbol, "1d", "1mo")
    closes = _parse_chart_closes(data)

    if len(closes) < 2:
        raise ValueError(f"Not enough price data for {symbol}")

    current = round(float(closes[-1]), 2)
    prev    = round(float(closes[-2]), 2)
    change  = round(current - prev, 2)
    pct     = round((change / prev) * 100, 2) if prev else 0.0
    return current, change, pct


def fetch_yahoo_historical(symbol, period):
    """Fetch full OHLCV history for a symbol over the given period."""
    range_map = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y":  "1y",  "2y":  "2y",  "5y":  "5y",
    }
    yf_range = range_map.get(period, "6mo")
    data = _yahoo_chart_request(symbol, "1d", yf_range)
    return _parse_chart_records(data)


def fetch_live_ohlcv(symbol):
    """Fetch the most recent day's OHLCV — replaces yf.Ticker().history()."""
    data    = _yahoo_chart_request(symbol, "1d", "5d")
    records = _parse_chart_records(data)
    if not records:
        raise ValueError(f"No data for {symbol}")
    latest = records[-1]
    return latest["open"], latest["high"], latest["low"], latest["volume"], latest["close"]

# ──────────────────────────────────────────────────────────────
# STOCK LISTS
# ──────────────────────────────────────────────────────────────
TICKER_STOCKS = [
    ("NIFTY 50",   "^NSEI"),
    ("SENSEX",     "^BSESN"),
    ("TCS",        "TCS.NS"),
    ("RELIANCE",   "RELIANCE.NS"),
    ("INFY",       "INFY.NS"),
    ("HDFC BANK",  "HDFCBANK.NS"),
    ("ICICI BANK", "ICICIBANK.NS"),
    ("WIPRO",      "WIPRO.NS"),
    ("BHARTI",     "BHARTIARTL.NS"),
    ("ITC",        "ITC.NS"),
    ("BAJFINANCE", "BAJFINANCE.NS"),
    ("SBIN",       "SBIN.NS"),
]

NIFTY50_SYMBOLS = [
    ("RELIANCE",   "RELIANCE.NS"),
    ("TCS",        "TCS.NS"),
    ("HDFCBANK",   "HDFCBANK.NS"),
    ("INFY",       "INFY.NS"),
    ("ICICIBANK",  "ICICIBANK.NS"),
    ("HINDUNILVR", "HINDUNILVR.NS"),
    ("ITC",        "ITC.NS"),
    ("SBIN",       "SBIN.NS"),
    ("BHARTIARTL", "BHARTIARTL.NS"),
    ("KOTAKBANK",  "KOTAKBANK.NS"),
    ("LT",         "LT.NS"),
    ("BAJFINANCE", "BAJFINANCE.NS"),
    ("WIPRO",      "WIPRO.NS"),
    ("HCLTECH",    "HCLTECH.NS"),
    ("AXISBANK",   "AXISBANK.NS"),
    ("ASIANPAINT", "ASIANPAINT.NS"),
    ("MARUTI",     "MARUTI.NS"),
    ("SUNPHARMA",  "SUNPHARMA.NS"),
    ("TATAMOTORS", "TATAMOTORS.NS"),
    ("NTPC",       "NTPC.NS"),
    ("POWERGRID",  "POWERGRID.NS"),
    ("ULTRACEMCO", "ULTRACEMCO.NS"),
    ("TITAN",      "TITAN.NS"),
    ("NESTLEIND",  "NESTLEIND.NS"),
    ("TECHM",      "TECHM.NS"),
]

HISTORICAL_TICKERS = {
    "NIFTY":    "^NSEI",
    "SENSEX":   "^BSESN",
    "TCS":      "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "INFY":     "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "WIPRO":    "WIPRO.NS",
    "SBIN":     "SBIN.NS",
    "MARUTI":   "MARUTI.NS",
}

# ──────────────────────────────────────────────────────────────
# PAGE ROUTES (from server.py)
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/holiday')
def holiday():
    return render_template('holiday.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/press_release')
def press_release():
    return render_template('press_release.html')

# ── Pattern pages ──────────────────────────────────────────────
@app.route('/reversal_chart_pattern')
def reversal_chart_pattern():
    return render_template('reversal_chart_pattern.html')

@app.route('/sidewase_chart_pattern')
def sideways_chart_pattern():
    return render_template('sidewase_chart_pattern.html')

@app.route('/single_candlestick')
def single_candlestick():
    return render_template('single_candlestick.html')

@app.route('/double_candelestic')
def double_candlestick():
    return render_template('double_candelestic.html')

@app.route('/triple_candelestic')
def triple_candlestick():
    return render_template('triple_candelestic.html')

@app.route('/continuation_chart_pattern')
def continuation_chart_pattern():
    return render_template('continuation_chart_pattern.html')

# ──────────────────────────────────────────────────────────────
# AUTH ROUTES
# ──────────────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register():
    data     = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username:
            return jsonify({"success": False, "message": "User already exists"})

    users.append({"username": username, "password": password})
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

    return jsonify({"success": True, "message": "Registered successfully"})


@app.route('/login', methods=['POST'])
def login():
    data     = request.get_json()
    username = data.get("username")
    password = data.get("password")

    with open(USER_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["username"] == username and user["password"] == password:
            return jsonify({"success": True, "message": "Login successful"})

    return jsonify({"success": False, "message": "Invalid credentials"})


@app.route('/contact-submit', methods=['POST'])
def contact_submit():
    data = request.get_json()
    with open(CONTACT_FILE, "r") as f:
        contacts = json.load(f)
    contacts.append(data)
    with open(CONTACT_FILE, "w") as f:
        json.dump(contacts, f)
    return jsonify({"success": True, "message": "Message saved successfully"})

# ──────────────────────────────────────────────────────────────
# API ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/test")
def test():
    return jsonify({"success": True, "message": "API is working 🚀"})


# ── Manual Prediction ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        company    = body.get("company")
        open_price = float(body.get("open"))
        high_price = float(body.get("high"))
        low_price  = float(body.get("low"))
        volume     = float(body.get("volume"))

        model_key_map = {
            "TCS":      "tcs",
            "Reliance": "reliance",
            "NIFTY":    "nifty50",
            "SENSEX":   "sensex",
        }
        key = model_key_map.get(company)
        if not key:
            return jsonify({"success": False, "error": f"Unknown company: {company}"}), 400
        if key not in models:
            return jsonify({"success": False, "error": "Model not found"}), 500

        model      = models[key]
        input_data = np.array([[open_price, high_price, low_price, volume]])
        prediction = model.predict(input_data)[0]

        return jsonify({
            "success":         True,
            "company":         company,
            "predicted_close": float(prediction),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Live + LSTM Prediction ─────────────────────────────────────
@app.route("/predict_live", methods=["GET"])
def predict_live():
    try:
        company = request.args.get("company")

        ticker_map = {
            "TCS":      "TCS.NS",
            "Reliance": "RELIANCE.NS",
            "NIFTY":    "^NSEI",
            "SENSEX":   "^BSESN",
        }
        model_key_map = {
            "TCS":      "tcs",
            "Reliance": "reliance",
            "NIFTY":    "nifty50",
            "SENSEX":   "sensex",
        }

        ticker = ticker_map.get(company)
        if not ticker:
            return jsonify({"success": False, "error": "Unknown company"}), 400

        # Fetch live OHLCV via our Yahoo session (no yfinance library needed)
        open_price, high_price, low_price, volume, close_price = fetch_live_ohlcv(ticker)

        # Ensemble model prediction
        key   = model_key_map[company]
        if key not in models:
            return jsonify({"success": False, "error": "Model not loaded"}), 500
        model = models[key]
        input_data = np.array([[open_price, high_price, low_price, volume]])
        price_pred = float(model.predict(input_data)[0])

        # LSTM prediction
        lstm_pred = None
        trend     = None
        if key in lstm_models:
            records    = fetch_yahoo_historical(ticker, "6mo")
            lstm_pred  = lstm_predict(records, lstm_models[key])
            if lstm_pred:
                trend = "UP" if lstm_pred > close_price else "DOWN"

        return jsonify({
            "success": True,
            "company": company,
            "live_data": {
                "open":   float(open_price),
                "high":   float(high_price),
                "low":    float(low_price),
                "volume": float(volume),
            },
            "predicted_close": price_pred,
            "lstm_prediction": lstm_pred,
            "trend":           trend,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Live Ticker ────────────────────────────────────────────────
@app.route("/live_ticker")
def live_ticker():
    results = []
    for name, symbol in TICKER_STOCKS:
        try:
            price, change, pct = fetch_yahoo_price(symbol)
            results.append({"name": name, "price": price, "change": change, "percent": pct})
            time.sleep(0.1)
        except Exception as e:
            print(f"[ticker] Skipping {symbol}: {e}")
            continue

    if not results:
        return jsonify({"success": False, "error": "Yahoo Finance unavailable. Try again in a minute."}), 500

    return jsonify({"success": True, "data": results})


# ── Top Gainers ────────────────────────────────────────────────
@app.route("/top_gainers")
def top_gainers():
    try:
        gainers = []
        for name, symbol in NIFTY50_SYMBOLS:
            try:
                price, change, pct = fetch_yahoo_price(symbol)
                if pct > 0:
                    gainers.append({"name": name, "price": price, "change": change, "percent": pct})
                time.sleep(0.08)
            except Exception as e:
                print(f"[gainers] Skipping {symbol}: {e}")
                continue

        gainers.sort(key=lambda x: x["percent"], reverse=True)
        top5 = gainers[:5]

        if not top5:
            return jsonify({"success": False, "error": "No gainers data. Market may be closed."}), 500

        return jsonify({"success": True, "data": top5})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Historical Data ────────────────────────────────────────────
@app.route("/historical")
def historical():
    try:
        symbol_key = request.args.get("symbol", "NIFTY")
        period     = request.args.get("period", "6mo")

        ticker_sym = HISTORICAL_TICKERS.get(symbol_key)
        if not ticker_sym:
            return jsonify({"success": False, "error": f"Unknown symbol: {symbol_key}"}), 400

        records = fetch_yahoo_historical(ticker_sym, period)

        if not records:
            return jsonify({"success": False, "error": "No data returned — market may be closed or symbol invalid"}), 404

        closes      = [r["close"]  for r in records]
        first_close = closes[0]
        last_close  = closes[-1]
        pct_change  = round(((last_close - first_close) / first_close) * 100, 2) if first_close else 0
        period_high = round(max(r["high"]   for r in records), 2)
        period_low  = round(min(r["low"]    for r in records), 2)
        avg_vol     = int(sum(r["volume"]   for r in records) / len(records))

        return jsonify({
            "success": True,
            "symbol":  symbol_key,
            "period":  period,
            "data":    records,
            "summary": {
                "start_price": first_close,
                "end_price":   last_close,
                "pct_change":  pct_change,
                "period_high": period_high,
                "period_low":  period_low,
                "avg_volume":  avg_vol,
                "data_points": len(records),
            },
        })

    except Exception as e:
        import traceback
        print("[historical] Error:", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500



# ──────────────────────────────────────────────────────────────
# DEBUG ENDPOINT — visit /debug in browser to diagnose issues
# REMOVE THIS ROUTE before going to production
# ──────────────────────────────────────────────────────────────
@app.route("/debug")
def debug():
    import traceback

    report = {
        "status":        "running",
        "base_dir":      BASE_DIR,
        "model_dir":     MODEL_DIR,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "pkl_models_loaded":  list(models.keys()),
        "lstm_models_loaded": list(lstm_models.keys()),
        "tensorflow_available": TF_AVAILABLE,
        "yahoo_crumb_present":  bool(_yahoo_crumb),
        "yahoo_session_present": bool(_yahoo_session),
    }

    # List files in models/ dir
    if os.path.exists(MODEL_DIR):
        report["model_dir_files"] = os.listdir(MODEL_DIR)
    else:
        report["model_dir_files"] = "DIRECTORY NOT FOUND"

    # Test Yahoo Finance connectivity
    yahoo_test = {}
    try:
        price, change, pct = fetch_yahoo_price("TCS.NS")
        yahoo_test["status"] = "OK"
        yahoo_test["tcs_price"] = price
    except Exception as e:
        yahoo_test["status"] = "FAILED"
        yahoo_test["error"] = str(e)
        yahoo_test["traceback"] = traceback.format_exc()
    report["yahoo_test"] = yahoo_test

    # Test model prediction
    model_test = {}
    if "tcs" in models:
        try:
            dummy = np.array([[3500.0, 3600.0, 3450.0, 1000000.0]])
            pred  = float(models["tcs"].predict(dummy)[0])
            model_test["status"] = "OK"
            model_test["tcs_dummy_pred"] = pred
        except Exception as e:
            model_test["status"] = "FAILED"
            model_test["error"] = str(e)
    else:
        model_test["status"] = "NO MODEL LOADED"
        model_test["hint"] = "Check model_dir_files above — .pkl files must be inside models/ folder"
    report["model_test"] = model_test

    return jsonify(report)


# ──────────────────────────────────────────────────────────────
# STARTUP & RUN
# ──────────────────────────────────────────────────────────────
# Warm up Yahoo session when the app starts (runs at deploy time)
with app.app_context():
    try:
        init_yahoo_session(force=True)
    except Exception as e:
        print(f"[startup] Yahoo session init warning: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
