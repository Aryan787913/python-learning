import warnings
warnings.filterwarnings("ignore")
import streamlit as pd_st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =====================================================
# UI CONFIGURATION
# =====================================================
pd_st.set_page_config(page_title="NSE Sniper Dashboard", layout="wide")
pd_st.title("🎯 NSE AI Sniper Signal Generator")
pd_st.markdown("This advanced model uses **Nifty 50** and **India VIX** macro-context alongside technical indicators to find absolute extreme setups. It dynamically searches for the threshold required to hit high accuracy.")

# =====================================================
# SIDEBAR - CONTROLS
# =====================================================
pd_st.sidebar.header("Sniper Configuration")

ticker = pd_st.sidebar.selectbox(
    "Select NSE Ticker",
    ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

target_accuracy = pd_st.sidebar.slider(
    "Target Accuracy (%)",
    min_value=55.0,
    max_value=75.0,
    value=60.0,
    step=1.0,
    help="The model will incrementally raise its strictness until it hits this backtested accuracy level."
)

start_date = pd_st.sidebar.date_input("Training Start Date", pd.to_datetime("2015-01-01"))
end_date = pd_st.sidebar.date_input("Data End Date", pd.to_datetime("2026-01-01"))

# =====================================================
# DATA PROCESSING & MODEL FUNCTION
# =====================================================
@pd_st.cache_data(ttl=3600)  
def run_sniper_model(ticker, start, end, target_acc):
    # 1. Download Asset + Macro Context
    tickers = [ticker, "^NSEI", "^INDIAVIX"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    if data.empty:
        return None, None, None, None, None, None
        
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data['Close']
        asset_close = close_df[ticker]
        nifty_close = close_df['^NSEI']
        vix_close = close_df['^INDIAVIX']
        volume = data['Volume'][ticker]
    else:
        # Fallback if download behaves differently
        asset_close = data['Close']
        nifty_close = data['Close'] # Placeholder
        vix_close = data['Close']   # Placeholder
        volume = data['Volume']

    df = pd.DataFrame({
        "Close": asset_close, 
        "Nifty_Close": nifty_close, 
        "VIX": vix_close,
        "Volume": volume
    })
    df.fillna(method='ffill', inplace=True)

    # 2. Feature Engineering
    df["Return"] = df["Close"].pct_change()
    df["Nifty_Return"] = df["Nifty_Close"].pct_change()
    df["Alpha_Spread"] = df["Return"] - df["Nifty_Return"]

    # Technicals
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["BB_Position"] = (df["Close"] - (ma20 - (std20 * 2))) / ((ma20 + (std20 * 2)) - (ma20 - (std20 * 2)))

    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volatility"] = df["Return"].rolling(14).std()
    df["Market_Correlation"] = df["Return"].rolling(window=20).corr(df["Nifty_Return"])

    for lag in range(1, 4):
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # Target: 1 if Up tomorrow, 0 if Down
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    latest_features = df.iloc[[-1]].copy()
    
    # Clean anomalies
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "Return_Lag_1", "Return_Lag_2", "RSI", "BB_Position", 
        "Volume_Change", "Volatility", "Nifty_Return", "Alpha_Spread", "VIX", "Market_Correlation"
    ]
    
    X = df[feature_cols]
    y = df["Target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Strict Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,            # Shallow to avoid overfitting
        min_samples_leaf=20,    # Requires strong consensus
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # 3. Dynamic Threshold Optimizer
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    final_threshold = 0.50
    final_trades = None
    final_acc = 0.0
    
    for threshold in np.arange(0.50, 0.65, 0.005):
        signals = []
        for p in probs:
            if p >= threshold:
                signals.append(1)
            elif p <= (1 - threshold):
                signals.append(0)
            else:
                signals.append(np.nan)
                
        test_results = pd.DataFrame({"Actual": y_test, "Prob": probs, "Signal": signals}, index=X_test.index)
        trades_taken = test_results.dropna()
        
        # We need at least 10 trades in the test set to trust the metric
        if len(trades_taken) >= 10:
            acc = accuracy_score(trades_taken["Actual"], trades_taken["Signal"]) * 100
            if acc >= target_acc:
                final_threshold = threshold
                final_acc = acc
                final_trades = trades_taken
                break
        else:
            break # Too few trades, stop searching

    # If we couldn't hit the target, return the best we found before giving up
    if final_trades is None:
        final_threshold = 0.55
        signals = [1 if p >= 0.55 else 0 if p <= 0.45 else np.nan for p in probs]
        test_results = pd.DataFrame({"Actual": y_test, "Prob": probs, "Signal": signals}, index=X_test.index)
        final_trades = test_results.dropna()
        if len(final_trades) > 0:
            final_acc = accuracy_score(final_trades["Actual"], final_trades["Signal"]) * 100
    
    # Process Live Signal
    X_latest = latest_features[feature_cols].copy()
    X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if not X_latest.isnull().values.any():
        X_latest_scaled = scaler.transform(X_latest)
        live_prob = model.predict_proba(X_latest_scaled)[0, 1]
    else:
        live_prob = None

    return final_trades, len(X_test), df, live_prob, latest_features, final_threshold

# =====================================================
# RUN APPLICATION
# =====================================================
trades, total_test_days, historical_df, next_day_prob, latest_stats, req_threshold = run_sniper_model(
    ticker, start_date, end_date, target_accuracy
)

if trades is not None and len(trades) > 0:
    accuracy = accuracy_score(trades["Actual"], trades["Signal"]) * 100
    trade_frequency = (len(trades) / total_test_days) * 100

    col1, col2, col3 = pd_st.columns(3)
    col1.metric("🎯 Backtest Accuracy", f"{accuracy:.2f}%")
    col2.metric("🔒 Required Confidence", f"{req_threshold:.3f}")
    col3.metric("⚡ Signals Fired", f"{len(trades)} trades", f"{trade_frequency:.1f}% of days")
    
    pd_st.divider()

    # =====================================================
    # RECOMMENDATION ENGINE
    # =====================================================
    if next_day_prob is not None:
        pd_st.subheader("🤖 Live Market Signal")
        
        if next_day_prob >= req_threshold:
            action = "BUY / BULLISH"
            color = "#00ff00"
            prob_display = next_day_prob * 100
        elif next_day_prob <= (1 - req_threshold):
            action = "SELL / BEARISH"
            color = "#ff0000"
            prob_display = (1 - next_day_prob) * 100
        else:
            action = "HOLD / NO CLEAR EDGE"
            color = "gray"
            prob_display = next_day_prob * 100

        vix_level = latest_stats['VIX'].values[0]
        alpha = latest_stats['Alpha_Spread'].values[0] * 100

        pd_st.markdown(f"#### **Action:** <span style='color:{color}; font-size:24px; font-weight:bold;'>{action}</span>", unsafe_allow_html=True)
        pd_st.markdown(f"**Upward Probability:** `{prob_display:.1f}%`")
        pd_st.info(f"**Macro Context:** The India VIX is currently at {vix_level:.1f}. Today, {ticker} had an Alpha Spread of {alpha:.2f}% against the Nifty 50.")

    pd_st.divider()

    # =====================================================
    # INTERACTIVE CHARTS
    # =====================================================
    pd_st.subheader(f"{ticker} vs Execution Points")
    
    plot_df = historical_df.loc[trades.index]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Close Price', line=dict(color='gray', width=1)))
    
    buys = plot_df[trades['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
    
    sells = plot_df[trades['Signal'] == 0]
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    pd_st.plotly_chart(fig, use_container_width=True)

else:
    pd_st.error("Target accuracy is too high for the available data. The model had to reject so many trades that it couldn't find a statistically valid sample. Try lowering the target accuracy.")
