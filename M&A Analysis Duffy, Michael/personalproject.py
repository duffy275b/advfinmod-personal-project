import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="M&A Deals and Comparable Company Analysis", layout="wide")

DEALS_CSV = "ma_deals.csv"

def load_deals():
    if os.path.exists(DEALS_CSV):
        return pd.read_csv(DEALS_CSV).to_dict('records')
    else:
        return [
            {"date": "2019-07-25", "target": "Intel-Smartphone modem bus.", "deal_value": 1.0},
            {"date": "2020-05-14", "target": "NextVR Inc", "deal_value": 0.0},
            {"date": "2023-03-27", "target": "WaveOne Inc", "deal_value": 0.0},
        ]

def save_deals(deals):
    df = pd.DataFrame(deals)
    df.to_csv(DEALS_CSV, index=False)

if "ma_deals" not in st.session_state:
    st.session_state.ma_deals = load_deals()

if "peers" not in st.session_state:
    st.session_state.peers = ["MSFT", "GOOGL", "AMZN"]

def fetch_stock_data(ticker, start="2018-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [" ".join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

    adj_close_cols = [c for c in data.columns if "Adj Close" in c]
    if adj_close_cols:
        close_col = adj_close_cols[0]
    else:
        close_col = "Close" if "Close" in data.columns else None

    if close_col is None or close_col not in data.columns:
        return pd.DataFrame(columns=["Date","Close"])

    data = data[[close_col]].dropna()
    data.reset_index(inplace=True)
    data.rename(columns={close_col: "Close"}, inplace=True)
    if 'Date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = data['Date'].dt.tz_localize(None)
    return data

def calculate_stock_price_change(data, deal_date):
    if data.empty:
        return np.nan
    pre_mask = data['Date'] < deal_date
    post_mask = data['Date'] > deal_date
    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        return np.nan
    pre_price = data.loc[pre_mask, 'Close'].mean()
    post_price = data.loc[post_mask, 'Close'].mean()
    return ((post_price - pre_price) / pre_price) * 100 if pre_price != 0 else np.nan

# NEW: Fetch financial statements and key metrics
def fetch_financial_metrics(ticker):
    # Attempt to fetch financials using yfinance
    t = yf.Ticker(ticker)
    # financials: Income statement (annual)
    try:
        fin = t.financials
        # Columns are dates, rows are line items.
        # For example fin.loc['Total Revenue'] might give multiple columns of data by year.
        if fin.empty:
            return {}

        # Sort columns by date (most recent first)
        fin = fin.sort_index(axis=1, ascending=False)

        # Extract latest and previous period data if available
        periods = fin.columns
        latest_period = periods[0]
        prev_period = periods[1] if len(periods) > 1 else None

        revenue_latest = fin.loc['Total Revenue', latest_period] if 'Total Revenue' in fin.index else np.nan
        revenue_prev = fin.loc['Total Revenue', prev_period] if prev_period and 'Total Revenue' in fin.index else np.nan
        operating_income = fin.loc['Operating Income', latest_period] if 'Operating Income' in fin.index else np.nan
        net_income = fin.loc['Net Income', latest_period] if 'Net Income' in fin.index else np.nan

        # Compute metrics
        revenue_growth = np.nan
        if not np.isnan(revenue_latest) and not np.isnan(revenue_prev) and revenue_prev != 0:
            revenue_growth = ((revenue_latest - revenue_prev) / revenue_prev) * 100

        operating_margin = np.nan
        if not np.isnan(operating_income) and not np.isnan(revenue_latest) and revenue_latest != 0:
            operating_margin = (operating_income / revenue_latest) * 100

        net_income_margin = np.nan
        if not np.isnan(net_income) and not np.isnan(revenue_latest) and revenue_latest != 0:
            net_income_margin = (net_income / revenue_latest) * 100

        # Market Cap
        # Attempt to fetch market cap from fast_info if available
        market_cap = t.fast_info.get('market_cap', np.nan)

        metrics = {
            "revenue_growth": revenue_growth,
            "operating_margin": operating_margin,
            "net_income_margin": net_income_margin,
            "market_cap": market_cap,
        }
        return metrics
    except Exception as e:
        return {}

st.title("M&A Deals and Comparable Company Analysis")

st.sidebar.header("Configure Analysis")
acquirer_ticker = st.sidebar.text_input("Acquirer Ticker", value="AAPL")

st.sidebar.subheader("Manage Peers")
new_peer = st.sidebar.text_input("New Peer Ticker", value="")
add_peer_btn = st.sidebar.button("Add Peer")
if add_peer_btn:
    peer = new_peer.strip().upper()
    if peer and peer not in st.session_state.peers:
        st.session_state.peers.append(peer)
        st.success(f"Peer {peer} added.")
    elif peer in st.session_state.peers:
        st.warning(f"Peer {peer} already in list.")

if st.session_state.peers:
    remove_peer = st.sidebar.selectbox("Select Peer to Remove", options=[""]+st.session_state.peers)
    remove_peer_btn = st.sidebar.button("Remove Peer")
    if remove_peer_btn and remove_peer != "":
        st.session_state.peers = [p for p in st.session_state.peers if p != remove_peer]
        st.success(f"Peer {remove_peer} removed.")
else:
    st.info("No peers selected yet. Add peers above.")

peer_tickers = st.session_state.peers

# Manage M&A deals
st.sidebar.subheader("Manage M&A Deals")
deal_date_input = st.sidebar.text_input("Deal Date (YYYY-MM-DD)", value="")
deal_target_input = st.sidebar.text_input("Target Name", value="")
deal_value_input = st.sidebar.number_input("Deal Value (in billions)", value=0.0, step=0.1)
add_deal_btn = st.sidebar.button("Add Deal")
remove_deal = st.sidebar.selectbox("Select Deal to Remove", options=[""]+[d["date"] for d in st.session_state.ma_deals])
remove_deal_btn = st.sidebar.button("Remove Deal")

if add_deal_btn:
    try:
        datetime.strptime(deal_date_input, '%Y-%m-%d')
        if any(d['date'] == deal_date_input for d in st.session_state.ma_deals):
            st.warning("A deal with this date already exists.")
        else:
            st.session_state.ma_deals.append({
                "date": deal_date_input,
                "target": deal_target_input,
                "deal_value": deal_value_input
            })
            save_deals(st.session_state.ma_deals)
            st.success("Deal added successfully!")
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")

if remove_deal_btn and remove_deal != "":
    st.session_state.ma_deals = [d for d in st.session_state.ma_deals if d["date"] != remove_deal]
    save_deals(st.session_state.ma_deals)
    st.success("Deal removed successfully!")

st.subheader("Current M&A Deals")
ma_df = pd.DataFrame(st.session_state.ma_deals)
st.dataframe(ma_df)

if len(ma_df) == 0:
    st.warning("No deals available. Add a deal in the sidebar.")
    st.stop()

selected_deal = st.selectbox("Select an M&A Deal for Analysis", options=ma_df['date'].tolist())
deal_info = ma_df[ma_df['date'] == selected_deal].iloc[0]
deal_date = pd.to_datetime(deal_info['date'])

acquirer_data = fetch_stock_data(acquirer_ticker)
if acquirer_data.empty:
    st.warning(f"No data found for {acquirer_ticker}. Check if the ticker is correct or try a different date range.")
peer_data = {pt: fetch_stock_data(pt) for pt in peer_tickers}

window = 90
start_date = deal_date - timedelta(days=window)
end_date = deal_date + timedelta(days=window)
acquirer_subset = acquirer_data[(acquirer_data['Date'] >= start_date) & (acquirer_data['Date'] <= end_date)]

fig_stock = go.Figure()
if not acquirer_subset.empty:
    fig_stock.add_trace(go.Scatter(x=acquirer_subset['Date'], y=acquirer_subset['Close'],
                                   mode='lines', name=acquirer_ticker))
else:
    st.warning("No acquirer stock data in the selected window.")

fig_stock.add_vline(x=deal_date, line_dash="dash", line_color="red", name="Deal Date")
fig_stock.update_layout(
    title=f"{acquirer_ticker} Stock Price Around {deal_info['target']} Deal",
    xaxis_title="Date",
    yaxis_title="Stock Price (USD)",
    template="plotly_white"
)
st.plotly_chart(fig_stock, use_container_width=True)

acquirer_change = calculate_stock_price_change(acquirer_data, deal_date)
metrics = [{"ticker": acquirer_ticker, "price_change": acquirer_change, "type": "Acquirer"}]

for p, p_data in peer_data.items():
    if p_data.empty:
        st.warning(f"No data for peer ticker {p}.")
    peer_change = calculate_stock_price_change(p_data, deal_date)
    metrics.append({"ticker": p, "price_change": peer_change, "type": "Peer"})

metrics_df = pd.DataFrame(metrics)
metrics_df['price_change'] = pd.to_numeric(metrics_df['price_change'], errors='coerce')

fig_peers = px.bar(metrics_df, x='ticker', y='price_change', color='type',
                   title="Price Change Comparison: Acquirer vs. Peers",
                   labels={'price_change': '% Price Change', 'ticker': 'Company'})
fig_peers.update_layout(template="plotly_white")
st.plotly_chart(fig_peers, use_container_width=True)

st.subheader(f"Analysis for {deal_info['target']}")
st.write(f"**Date Announced:** {deal_date.date()}")
st.write(f"**Deal Value:** ${deal_info['deal_value']} billion")

acq_change_val = metrics_df.loc[metrics_df['ticker'] == acquirer_ticker, 'price_change'].values[0]
peer_avg = metrics_df[metrics_df['type'] == 'Peer']['price_change'].mean()

st.write("**Acquirer vs. Peers Performance:**")
if np.isnan(acq_change_val):
    st.write("Acquirer Price Change: N/A")
else:
    st.write(f"Acquirer Price Change: {acq_change_val:.2f}%")

if np.isnan(peer_avg):
    st.write("Peer Average Change: N/A")
else:
    st.write(f"Peer Average Change: {peer_avg:.2f}%")

# NEW: Additional M&A-Related Financial Metrics from actual data
st.header("Additional M&A Financial Metrics")

acquirer_metrics = fetch_financial_metrics(acquirer_ticker)

if not acquirer_metrics:
    st.write("No financial data available for additional metrics.")
else:
    rev_growth = acquirer_metrics.get("revenue_growth", np.nan)
    op_margin = acquirer_metrics.get("operating_margin", np.nan)
    ni_margin = acquirer_metrics.get("net_income_margin", np.nan)
    market_cap = acquirer_metrics.get("market_cap", np.nan)

    # Display the metrics if they are not NaN
    if not np.isnan(rev_growth):
        st.write(f"Revenue Growth (YoY): {rev_growth:.2f}%")
    else:
        st.write("Revenue Growth: N/A")

    if not np.isnan(op_margin):
        st.write(f"Operating Margin (Latest): {op_margin:.2f}%")
    else:
        st.write("Operating Margin: N/A")

    if not np.isnan(ni_margin):
        st.write(f"Net Income Margin (Latest): {ni_margin:.2f}%")
    else:
        st.write("Net Income Margin: N/A")

    if not np.isnan(market_cap):
        st.write(f"Market Capitalization: ${market_cap:,}")
    else:
        st.write("Market Capitalization: N/A")

# FINAL EVALUATION
st.header("Final Evaluation")

# Criteria for evaluation:
# 1. Positive if revenue growth > 0
# 2. Positive if net income margin > 0
# 3. Positive if acquirer price change >= peer average (or peer average is N/A)
# If all good, say "Overall positive outlook", else "Mixed or negative outlook".

positive_signs = 0
criteria_count = 0

# Check revenue growth
if not np.isnan(rev_growth):
    criteria_count += 1
    if rev_growth > 0:
        positive_signs += 1

# Check net income margin
if not np.isnan(ni_margin):
    criteria_count += 1
    if ni_margin > 0:
        positive_signs += 1

# Check acquirer vs peers
if not np.isnan(acq_change_val) and not np.isnan(peer_avg):
    criteria_count += 1
    if acq_change_val >= peer_avg:
        positive_signs += 1
elif not np.isnan(acq_change_val):
    # If no peer data, just count this as neutral or positive if acq is positive
    criteria_count += 1
    if acq_change_val > 0:
        positive_signs += 1

if criteria_count == 0:
    st.write("No sufficient metrics to form a conclusion.")
else:
    if positive_signs == criteria_count:
        st.write("**Overall Evaluation:** Positive")
    elif positive_signs >= criteria_count/2:
        st.write("**Overall Evaluation:** Mixed but leaning positive")
    else:
        st.write("**Overall Evaluation:** Negative or mixed")

st.write("These conclusions are based solely on available financial and stock performance metrics.")
