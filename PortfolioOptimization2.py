import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Portfolio Optimizer & Goal-Based Projection", layout="wide")
st.title("📈 Portfolio Optimization & Goal-Based Projection Dashboard")
st.write(
    """
    This tool:
    1) Builds an optimized portfolio using historical performance (max Sharpe),
    2) Runs Monte Carlo simulations to project future portfolio values,
    3) Estimates the probability of reaching your target wealth goal,
    4) Communicates risk via outcome bands (beyond standard deviation).
    """
)

# =========================
# Helpers
# =========================
def prettify_date_axis(ax, minticks=5, maxticks=8, rotation=0):
    """Clean, evenly-spaced date ticks with concise labels."""
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if rotation:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha("center")


def get_sample_prices(tickers, start, end, n_days=800, seed=42) -> pd.DataFrame:
    """Synthetic-but-reasonable price history for offline demo."""
    rng = np.random.default_rng(seed)
    all_days = pd.bdate_range(start=start, end=end)
    if len(all_days) > n_days:
        all_days = all_days[-n_days:]
    prices = {}
    for t in tickers:
        start_price = rng.uniform(50, 300)
        annual_drift = rng.uniform(0.05, 0.20)
        annual_vol = rng.uniform(0.20, 0.60)
        daily_drift = annual_drift / 252
        daily_vol = annual_vol / (252 ** 0.5)
        steps = rng.normal(daily_drift, daily_vol, size=len(all_days))
        path = np.empty(len(all_days))
        path[0] = start_price
        for i in range(1, len(all_days)):
            path[i] = path[i - 1] * (1 + steps[i])
        prices[t] = path
    return pd.DataFrame(prices, index=all_days)


@st.cache_data(show_spinner=False)
def _download_group_yf(tickers, start, end):
    """Single grouped yfinance call (may return MultiIndex columns)."""
    try:
        raw = yf.download(
            tickers,
            start=pd.to_datetime(start),
            end=pd.to_datetime(end),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        return raw
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _download_single_yf(ticker, start, end):
    """Single ticker download."""
    try:
        raw = yf.download(
            ticker,
            start=pd.to_datetime(start),
            end=pd.to_datetime(end),
            auto_adjust=True,
            progress=False,
        )
        return raw
    except Exception:
        return None


def _extract_close_from_raw(raw, tickers) -> pd.DataFrame:
    """
    Normalize yfinance result into a [date x ticker] close-price DataFrame.
    Works for both single- and multi-index returns.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    # Multi-index columns (typical for multi-ticker download)
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            close_df = raw["Close"].copy()
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame()
            cols = [c for c in tickers if c in close_df.columns]
            return close_df[cols].dropna(how="all") if cols else close_df.dropna(how="all")
        except Exception:
            close_cols = [c for c in raw.columns if isinstance(c, tuple) and c[0] == "Close"]
            if close_cols:
                tmp = raw[close_cols].copy()
                tmp.columns = [c[1] for c in close_cols]
                cols = [c for c in tickers if c in tmp.columns]
                return tmp[cols].dropna(how="all") if cols else tmp.dropna(how="all")
            return pd.DataFrame()

    # Single-level columns (single ticker)
    if "Close" in raw.columns:
        df = raw[["Close"]].copy()
    elif "Adj Close" in raw.columns:
        df = raw[["Adj Close"]].copy()
        df.columns = ["Close"]
    else:
        df = raw.iloc[:, :1].copy()
        df.columns = ["Close"]
    return df.dropna(how="all")


def _merge_single_ticker_frames(frames_dict: dict) -> pd.DataFrame:
    """Merge per-ticker close series into one matrix."""
    if not frames_dict:
        return pd.DataFrame()
    out = None
    for t, df in frames_dict.items():
        if df is None or df.empty:
            continue
        if "Close" not in df.columns:
            continue
        s = df["Close"].rename(t)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    return out.dropna(how="all") if out is not None else pd.DataFrame()


def _clean_dates_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def load_prices_yfinance(tickers, start, end):
    """
    Robust yfinance loader (per-ticker first):
    1) Try per-ticker downloads and merge into a matrix.
    2) If that fails or is empty, try a grouped download & extract Close.
    Returns (prices_df, debug_dict).
    """
    debug = {"phase": None, "group_attempt": None, "per_ticker_attempts": []}

    # Phase 1: Per-ticker download FIRST
    debug["phase"] = "per_ticker"
    frames = {}
    for t in tickers:
        raw_t = _download_single_yf(t, start, end)
        rows = int(len(raw_t)) if raw_t is not None else 0
        df_t = _extract_close_from_raw(raw_t, [t])
        frames[t] = df_t
        debug["per_ticker_attempts"].append(
            {
                "ticker": t,
                "raw_rows": rows,
                "ok": (df_t is not None and not df_t.empty),
            }
        )

    merged = _merge_single_ticker_frames(frames)
    merged = _clean_dates_window(merged, start, end)

    if merged is not None and not merged.empty:
        return merged, debug

    # Phase 2: Grouped download as fallback
    debug["phase"] = "group"
    group_raw = _download_group_yf(tickers, start, end)
    if group_raw is not None and len(group_raw) > 0:
        prices = _extract_close_from_raw(group_raw, tickers)
        prices = _clean_dates_window(prices, start, end)
        debug["group_attempt"] = {
            "rows": int(len(group_raw)),
            "extracted_cols": list(prices.columns) if prices is not None else [],
        }
        if prices is not None and not prices.empty:
            return prices, debug

    # If everything fails, return empty
    return pd.DataFrame(), debug


def get_data_wrds_placeholder(tickers, start, end):
    """Future work: WRDS/CRSP institutional data pipeline."""
    pass


def fmt_dollar(x):
    return f"${x:,.0f}"


def fmt_pct(x):
    return f"{x * 100:.2f}%"


# === Max Sharpe via true constrained optimization (SLSQP) ===
def solve_max_sharpe(annual_returns, annual_cov, rf_annual):
    """
    Maximize Sharpe = (w·mu - rf) / sqrt(wᵀ Σ w)
    s.t. sum(w) = 1, w >= 0
    """
    mu = annual_returns.values
    cov = annual_cov.values
    n = len(mu)

    def neg_sharpe(w):
        w = np.array(w)
        port_ret = float(np.dot(w, mu))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        if port_vol == 0:
            return 1e6
        sharpe = (port_ret - rf_annual) / port_vol
        return -sharpe  # minimize negative Sharpe

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.repeat(1.0 / n, n)

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if not result.success:
        w = w0
    else:
        w = result.x

    port_ret = float(np.dot(w, mu))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
    sharpe = (port_ret - rf_annual) / port_vol if port_vol > 0 else 0.0

    return w, port_ret, port_vol, sharpe


# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("📥 Portfolio Setup")
default_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
ticker_input = st.sidebar.text_input(
    "Enter stock tickers (comma-separated):", value=",".join(default_tickers)
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if not tickers:
    tickers = default_tickers

st.sidebar.subheader("Historical Data Window")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date", pd.Timestamp.today())

st.sidebar.subheader("Goal-Based Planning")
initial_investment = st.sidebar.number_input(
    "Starting portfolio value ($)",
    min_value=1000.0,
    value=10000.0,
    step=1000.0,
    format="%.2f",
)
target_goal = st.sidebar.number_input(
    "Target portfolio value at the end ($)",
    min_value=1000.0,
    value=50000.0,
    step=5000.0,
    format="%.2f",
)
years_to_goal = st.sidebar.slider(
    "Years until goal (Monte Carlo horizon)", min_value=1, max_value=30, value=10, step=1
)
n_simulations = st.sidebar.number_input(
    "Number of Monte Carlo simulations",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100,
)

st.sidebar.subheader("Risk-Free Rate")
risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.25,
)
rf_annual = risk_free_rate / 100.0
rf_daily = rf_annual / 252.0

st.sidebar.subheader("Backtest Settings")
use_out_of_sample = st.sidebar.checkbox(
    "Use out-of-sample backtest (train/test split)", value=True
)

show_raw_data = st.sidebar.checkbox("Show raw price data (debug)", value=False)

st.sidebar.write("---")
st.sidebar.caption(
    "Historical data via yfinance (Yahoo Finance). If retrieval fails (network/rate limits), "
    "the app falls back to synthetic data. Future versions can use WRDS/CRSP."
)

# =========================
# Load Data (yfinance → fallback)
# =========================
prices, yf_debug = load_prices_yfinance(tickers, start_date, end_date)
data_source = "yfinance"

if prices is None or prices.empty:
    prices = get_sample_prices(tickers, start_date, end_date)
    data_source = "sample (offline fallback)"

if prices is None or prices.empty:
    st.error("Could not load any price data (yfinance failed and fallback failed).")
    st.write("Tickers:", tickers)
    st.write("Start:", start_date, "End:", end_date)
    st.subheader("yfinance debug")
    st.json(yf_debug)
    st.stop()

if show_raw_data:
    st.subheader("🔎 Price Data Used (first 10 rows)")
    st.caption(f"Source: {data_source}")
    st.dataframe(prices.head(10))

if data_source != "yfinance":
    st.warning(
        "Using synthetic sample price data (live fetch unavailable). In production, use yfinance/WRDS."
    )
else:
    st.success("Using live yfinance data ✅")
    with st.expander("yfinance loader debug"):
        st.json(yf_debug)

# Compute returns
returns = prices.pct_change().dropna()
if returns.empty:
    st.error("Not enough historical data to compute returns. Try a wider date range.")
    st.stop()

# =========================
# Split into train / test for out-of-sample (if enabled)
# =========================
train_returns = returns
test_returns = returns
split_date = None

if use_out_of_sample:
    split_idx = int(len(returns) * 0.5)  # simple 50/50 split
    # require at least ~1 year of data in each side to be meaningful
    if split_idx < 252 or len(returns) - split_idx < 252:
        st.warning(
            "Not enough data for a 50/50 train/test split with at least 1 year on each side. "
            "Using full sample for both optimization and backtest (in-sample)."
        )
        use_out_of_sample = False
    else:
        train_returns = returns.iloc[:split_idx]
        test_returns = returns.iloc[split_idx:]
        split_date = test_returns.index[0]

# =========================
# 1) Portfolio Optimization (Max Sharpe, constrained)
# =========================
st.header("1. 📊 Portfolio Optimization (Max Sharpe Ratio)")

mean_daily_returns_train = train_returns.mean()
cov_daily_train = train_returns.cov()
annual_returns_train = mean_daily_returns_train * 252
annual_cov_train = cov_daily_train * 252

# True constrained optimization instead of random Monte Carlo weights
opt_weights, opt_return, opt_volatility, opt_sharpe = solve_max_sharpe(
    annual_returns_train, annual_cov_train, rf_annual
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("📌 Recommended Allocation (Max Sharpe)")
    st.table(
        pd.DataFrame(
            {"Ticker": tickers, "Weight (%)": np.round(opt_weights * 100, 2)}
        )
    )
with c2:
    st.subheader("📈 Portfolio Stats (Annualized, In-Sample)")
    st.metric("Expected Return", f"{opt_return * 100:.2f}%")
    st.metric("Volatility (Stdev)", f"{opt_volatility * 100:.2f}%")
    st.metric("Sharpe Ratio (excess over risk-free)", f"{opt_sharpe:.2f}")

if use_out_of_sample and split_date is not None:
    st.info(
        f"""
        **Out-of-Sample Design (Train/Test Split)**  
        - Weights are estimated using the **first half** of the sample (up to {split_date.date()}).  
        - Performance is evaluated on the **second half** of the sample (from {split_date.date()} onward).  
        This mitigates look-ahead bias because the optimization only uses information that would have been 
        available at the time.
        """
    )
else:
    st.info(
        """
        **Important Note on Look-Ahead Bias**  
        When optimization and performance are based on the same full sample, the portfolio construction
        uses information that would not have been known in real time. This creates *look-ahead bias*.  
        The out-of-sample train/test split option in the sidebar mitigates this by separating the estimation
        period (train) from the evaluation period (test).
        """
    )

# =========================
# 2) Backtest vs S&P 500 (test period if out-of-sample)
# =========================
st.header("2. 🏁 Backtest vs Benchmark (S&P 500)")

# Portfolio daily returns on the TEST period (if out-of-sample on, else full)
opt_daily = (test_returns * opt_weights).sum(axis=1)


def load_sp500_series(start, end):
    try:
        sp = yf.download(
            "^GSPC",
            start=pd.to_datetime(start),
            end=pd.to_datetime(end),
            auto_adjust=True,
            progress=False,
        )
        if sp is None or len(sp) == 0:
            return None
        if "Close" in sp.columns:
            s = sp["Close"]
        elif "Adj Close" in sp.columns:
            s = sp["Adj Close"]
        else:
            s = sp.iloc[:, 0]
        return s
    except Exception:
        return None


# Match S&P window to the backtest window actually used
sp_start = opt_daily.index[0]
sp_end = opt_daily.index[-1]
sp500_prices = load_sp500_series(sp_start, sp_end)

if sp500_prices is None or len(sp500_prices) == 0:
    # synthetic benchmark if S&P fetch fails
    bench_mean = opt_daily.mean() * 0.7
    bench_std = opt_daily.std() * 0.7
    rng_bench = np.random.default_rng(999)
    bench_random = rng_bench.normal(
        loc=bench_mean, scale=bench_std, size=len(opt_daily)
    )
    benchmark_returns = pd.Series(
        bench_random, index=opt_daily.index, name="S&P 500 (synthetic)"
    )
else:
    benchmark_returns = sp500_prices.pct_change().dropna()
    benchmark_returns = benchmark_returns.reindex(opt_daily.index).dropna()
    benchmark_returns.name = "S&P 500"

combined = pd.concat(
    [opt_daily.rename("Optimized Portfolio"), benchmark_returns], axis=1
).dropna()
cum_perf = (1 + combined).cumprod()

fig_perf, ax_perf = plt.subplots(figsize=(9, 4.5), dpi=120)
ax_perf.plot(
    cum_perf.index, cum_perf["Optimized Portfolio"], label="Optimized Portfolio"
)
ax_perf.plot(cum_perf.index, cum_perf.iloc[:, 1], label=cum_perf.columns[1])
ax_perf.set_title("Cumulative Growth of $1 (Backtest Period)")
ax_perf.set_ylabel("Growth of $1")
ax_perf.legend()

# Better date ticks
prettify_date_axis(ax_perf, minticks=5, maxticks=8)
fig_perf.tight_layout()
st.pyplot(fig_perf, use_container_width=True)

if use_out_of_sample and split_date is not None:
    st.caption("Backtest uses only the out-of-sample period (second half of the data).")
else:
    st.caption(
        "Backtest uses the full sample period (note the look-ahead bias caveat above)."
    )

# =========================
# 3) Monte Carlo Projection (future value, based on test distribution if out-of-sample)
# =========================
st.header("3. 🔮 Monte Carlo Projection of Future Portfolio Value")

tdpy = 252
total_days = years_to_goal * tdpy

# For Monte Carlo, follow professor's suggestion:
# - If out-of-sample is on, draw from the *test period* distribution.
# - Otherwise, use in-sample (optimization) stats.
if use_out_of_sample:
    port_daily_for_mc = opt_daily  # already test-period portfolio returns
else:
    port_daily_for_mc = (returns * opt_weights).sum(axis=1)

mc_annual_return = float(port_daily_for_mc.mean() * 252)
mc_annual_vol = float(port_daily_for_mc.std() * np.sqrt(252))

daily_mean = mc_annual_return / tdpy
daily_vol = mc_annual_vol / (tdpy ** 0.5)

paths = np.zeros((n_simulations, total_days + 1))
paths[:, 0] = initial_investment
rng_mc = np.random.default_rng(123)
for i in range(1, total_days + 1):
    rr = rng_mc.normal(loc=daily_mean, scale=daily_vol, size=n_simulations)
    paths[:, i] = paths[:, i - 1] * (1 + rr)

ending = paths[:, -1]
pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
p5, p25, p50, p75, p95 = pcts
prob_success = float((ending >= target_goal).mean() * 100.0)

cl, cr = st.columns([2, 1])
with cl:
    st.subheader("🚀 Projected Portfolio Value Over Time")
    years_axis = np.arange(total_days + 1) / tdpy
    fig_mc, ax_mc = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax_mc.fill_between(
        years_axis, p25, p75, alpha=0.3, label="50% band (25th–75th)"
    )
    ax_mc.plot(years_axis, p50, label="Median (50th)")
    ax_mc.plot(
        years_axis, p5, linestyle="--", alpha=0.5, label="5th (weak)"
    )
    ax_mc.plot(
        years_axis, p95, linestyle="--", alpha=0.5, label="95th (strong)"
    )
    ax_mc.set_xlabel("Years in the future")
    ax_mc.set_ylabel("Portfolio Value ($)")
    ax_mc.set_title("Simulated Future Portfolio Values")
    ax_mc.legend()

    # Whole-year ticks on x-axis
    ax_mc.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, prune="both"))

    fig_mc.tight_layout()
    st.pyplot(fig_mc, use_container_width=True)

    st.caption(
        "Monte Carlo uses the realized distribution of the portfolio returns "
        f"from the {'out-of-sample test period' if use_out_of_sample else 'full sample'}."
    )
with cr:
    st.subheader("🎯 Goal Check")
    st.metric(
        "Target at Horizon", fmt_dollar(target_goal), delta=f"{years_to_goal} year goal"
    )
    st.metric("Chance of Reaching Goal", f"{prob_success:.1f}%")
    w25, w50, w75 = np.percentile(ending, [25, 50, 75])
    st.write(
        f"- Weaker market (25th %ile): ~{fmt_dollar(w25)}\n"
        f"- Typical market (50th %ile): ~{fmt_dollar(w50)}\n"
        f"- Stronger market (75th %ile): ~{fmt_dollar(w75)}"
    )

# =========================
# 4) Scenario Summary
# =========================
st.header("4. 📌 Scenario Summary (Bull / Base / Bear)")
scenarios = pd.DataFrame(
    {
        "Scenario": ["Bear", "Base", "Bull"],
        "Assumed Annual Return": [
            opt_return - 0.05,
            opt_return,
            opt_return + 0.05,
        ],
        "Assumed Volatility": [
            max(0, opt_volatility + 0.05),
            opt_volatility,
            max(0, opt_volatility - 0.05),
        ],
    }
)
scenarios["Est. Value in " + str(years_to_goal) + "y"] = (
    initial_investment * (1 + scenarios["Assumed Annual Return"]) ** years_to_goal
)
st.table(
    scenarios.assign(
        **{
            "Assumed Annual Return": (scenarios["Assumed Annual Return"] * 100)
            .round(2)
            .astype(str)
            + "%",
            "Assumed Volatility": (scenarios["Assumed Volatility"] * 100)
            .round(2)
            .astype(str)
            + "%",
            "Est. Value in " + str(years_to_goal) + "y": scenarios[
                "Est. Value in " + str(years_to_goal) + "y"
            ].map(lambda x: fmt_dollar(x)),
        }
    )
)
st.caption(
    "Scenario view is a simple summary; Monte Carlo above is more realistic (full path simulation)."
)

# =========================
# 5) Final Summary
# =========================
st.header("5. 📝 Your Portfolio Performance Summary")
w25_end, w50_end, w75_end = np.percentile(ending, [25, 50, 75])
exp_ret_str = fmt_pct(opt_return)
vol_str = fmt_pct(opt_volatility)
sharpe_str = f"{opt_sharpe:.2f}"
initial_str = fmt_dollar(initial_investment)
goal_str = fmt_dollar(target_goal)
horizon_str = f"{years_to_goal} years"
prob_str = f"{prob_success:.1f}%"
w25_str, w50_str, w75_str = (
    fmt_dollar(w25_end),
    fmt_dollar(w50_end),
    fmt_dollar(w75_end),
)

sample_desc = "out-of-sample test period" if use_out_of_sample else "full sample period"

st.markdown(
    f"""
**Recommended Allocation (Max Sharpe, excess over risk-free)**  
- Expected annual return (in-sample): **{exp_ret_str}**  
- Annualized volatility (in-sample): **{vol_str}**  
- Sharpe ratio (using risk-free {risk_free_rate:.2f}%): **{sharpe_str}**

**Goal-Based Decision Support**  
- Starting value: **{initial_str}**  
- Target at horizon: **{goal_str}**  
- Horizon: **{horizon_str}**  
- Estimated probability of reaching target: **{prob_str}**

**Risk in Plain English (Ending Value Distribution)**  
- Weaker market (25th percentile): **{w25_str}**  
- Typical market (50th percentile / median): **{w50_str}**  
- Stronger market (75th percentile): **{w75_str}**

**Data & Methodology Notes**  
- Optimization sample (for weights and Sharpe): {"train half of the data" if use_out_of_sample else "entire historical window selected."}  
- Backtest + Monte Carlo distribution based on: **{sample_desc}**  
- Data source: **{data_source}**. If “sample”, live fetch was unavailable this run; in production, this would be replaced by yfinance.
"""
)