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
pd_st.set_page_config(page_title="NSE AI Signal Dashboard", layout="wide")
pd_st.title("📈 NSE AI Market Signal Generator")
pd_st.markdown("This application uses a Random Forest classifier paired with advanced technicals (MACD, Bollinger Bands, RSI) to generate directional signals.")

# =====================================================
# SIDEBAR - CONTROLS
# =====================================================
pd_st.sidebar.header("Model Configuration")

ticker = pd_st.sidebar.selectbox(
    "Select NSE Ticker",
    ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

confidence_threshold = pd_st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.65,
    value=0.54,
    step=0.01,
    help="Higher thresholds mean fewer, more precise signals."
)

start_date = pd_st.sidebar.date_input("Training Start Date", pd.to_datetime("2015-01-01"))
end_date = pd_st.sidebar.date_input("Data End Date", pd.to_datetime("2026-01-01"))

# =====================================================
# DATA PROCESSING & MODEL FUNCTION
# =====================================================
@pd_st.cache_data(ttl=3600)  
def run_trading_model(ticker, start, end, threshold):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        return None, None, None, None, None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Base Returns
    df["Return"] = df["Close"].pct_change()
    for lag in range(1, 4):
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # 2. RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    # 3. MACD Calculation
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands Position Calculation
    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = ma20 + (std20 * 2)
    df["Lower_Band"] = ma20 - (std20 * 2)
    # Position: 0 = at lower band, 1 = at upper band
    df["BB_Position"] = (df["Close"] - df["Lower_Band"]) / (df["Upper_Band"] - df["Lower_Band"])

    # 5. Volume and Volatility
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volatility"] = df["Return"].rolling(14).std()

    # Target: 1 if Up tomorrow, 0 if Down
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    # Save the absolute latest row for LIVE tomorrow prediction
    latest_features = df.iloc[[-1]].copy()
    
    # === THE FIX: Remove np.inf created by divisions by zero in technicals ===
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "Return_Lag_1", "Return_Lag_2", "Return_Lag_3", 
        "RSI", "MACD", "MACD_Signal", "BB_Position", 
        "Volume_Change", "Volatility"
    ]
    
    X = df[feature_cols]
    y = df["Target"]

    # Chronological Split (80% Train, 20% Test)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=15,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate Thresholds
    probs = model.predict_proba(X_test_scaled)[:, 1]
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
    
    # Process Live Signal for tomorrow
    X_latest = latest_features[feature_cols].copy()
    X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if not X_latest.isnull().values.any():
        X_latest_scaled = scaler.transform(X_latest)
        live_prob = model.predict_proba(X_latest_scaled)[0, 1]
    else:
        live_prob = None

    return trades_taken, len(X_test), df, live_prob, latest_features

# =====================================================
# RUN APPLICATION EXECUTION
# =====================================================
trades, total_test_days, historical_df, next_day_prob, latest_stats = run_trading_model(
    ticker, start_date, end_date, confidence_threshold
)

if trades is not None and len(trades) > 0:
    accuracy = accuracy_score(trades["Actual"], trades["Signal"]) * 100
    trade_frequency = (len(trades) / total_test_days) * 100

    col1, col2 = pd_st.columns(2)
    col1.metric("🎯 Directional Accuracy (Test Set)", f"{accuracy:.2f}%")
    col2.metric("⚡ Signals Fired", f"{len(trades)} days", f"{trade_frequency:.1f}% frequency")
    
    pd_st.divider()

    # =====================================================
    # INTELLIGENT RECOMMENDATION ENGINE
    # =====================================================
    if next_day_prob is not None:
        pd_st.subheader("🤖 Intelligent Recommendation for Tomorrow")
        
        # Determine strict signal logic
        if next_day_prob >= confidence_threshold:
            action = "BUY / BULLISH"
            color = "green"
            prob_display = next_day_prob * 100
        elif next_day_prob <= (1 - confidence_threshold):
            action = "SELL / BEARISH"
            color = "red"
            prob_display = (1 - next_day_prob) * 100
        else:
            action = "HOLD / SIDELINES"
            color = "gray"
            # If holding, we just show the raw upward probability
            prob_display = next_day_prob * 100

        # Read specific latest values safely
        current_rsi = latest_stats['RSI'].values[0]
        current_macd = latest_stats['MACD'].values[0]
        current_macd_signal = latest_stats['MACD_Signal'].values[0]
        current_bb = latest_stats['BB_Position'].values[0]

        # Generate automated reasoning
        reasons = []
        if current_rsi > 70:
            reasons.append(f"RSI is highly elevated ({current_rsi:.1f}), indicating the stock is technically overbought.")
        elif current_rsi < 30:
            reasons.append(f"RSI is suppressed ({current_rsi:.1f}), indicating the stock is technically oversold.")
        else:
            reasons.append(f"RSI is neutral at {current_rsi:.1f}.")
            
        if current_macd > current_macd_signal:
            reasons.append("MACD is demonstrating bullish momentum as it trends above its signal line.")
        else:
            reasons.append("MACD is demonstrating bearish momentum as it trends below its signal line.")
            
        if current_bb > 0.8:
            reasons.append("Price is heavily pressing against the upper Bollinger Band, which frequently acts as strong resistance.")
        elif current_bb < 0.2:
            reasons.append("Price is nearing the lower Bollinger Band, suggesting an approach to a statistical support floor.")
        else:
            reasons.append("Price is consolidating near the median of its Bollinger Bands.")

        reasoning_text = " ".join(reasons)

        pd_st.markdown(f"#### **AI Signal:** <span style='color:{color}; font-size:24px; font-weight:bold;'>{action}</span>", unsafe_allow_html=True)
        pd_st.markdown(f"**Mathematical Confidence:** `{prob_display:.1f}%`")
        pd_st.info(f"**Model Reasoning Insight:** {reasoning_text}")

    pd_st.divider()

    # =====================================================
    # INTERACTIVE CHARTS
    # =====================================================
    pd_st.subheader("Historical Stock Price vs AI Signal Execution")
    
    plot_df = historical_df.loc[trades.index]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Close Price', line=dict(color='gray', width=1)))
    
    buys = plot_df[trades['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='AI Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up')))
    
    sells = plot_df[trades['Signal'] == 0]
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='AI Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down')))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (INR)", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    pd_st.plotly_chart(fig, use_container_width=True)

else:
    pd_st.warning("No high-confidence signals generated for this specific configuration. Try lowering the threshold or widening the date range.")
