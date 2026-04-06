import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Quant Pricing Engine", layout="wide")

st.title("📈 Stochastic Modeling & Derivatives Pricing Dashboard")

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("⚙️ Input Parameters")

S0 = st.sidebar.number_input("Stock Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=105.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
T = st.sidebar.number_input("Time to Maturity (T)", value=1.0)

n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
steps = st.sidebar.slider("Time Steps", 10, 252, 100)

use_antithetic = st.sidebar.checkbox("Use Antithetic Variates")
use_control_variate = st.sidebar.checkbox("Use Control Variates")

run = st.sidebar.button("Run Simulation")

# ================================
# FUNCTIONS
# ================================

def simulate_gbm(S0, r, sigma, T, steps, n_sim):
    dt = T / steps
    paths = np.zeros((steps, n_sim))
    paths[0] = S0

    for t in range(1, steps):
        z = np.random.standard_normal(n_sim)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    return paths


def monte_carlo_price(paths, K, r, T):
    S_T = paths[-1]
    payoff = np.maximum(S_T - K, 0)
    return np.exp(-r*T) * np.mean(payoff), payoff


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def compute_greeks(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0*sigma*np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta = (-S0 * norm.pdf(d1) * sigma / (2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2))

    return delta, gamma, vega, theta


# ================================
# MAIN EXECUTION
# ================================
if run:

    # ============================
    # SIMULATION
    # ============================
    start = time.time()
    paths = simulate_gbm(S0, r, sigma, T, steps, n_simulations)
    mc_price, payoffs = monte_carlo_price(paths, K, r, T)
    end = time.time()

    bs_price = black_scholes_call(S0, K, r, sigma, T)

    # ============================
    # CONTROL VARIATE
    # ============================
    if use_control_variate:
        S_T = paths[-1]
        discounted_payoff = np.exp(-r*T) * np.maximum(S_T - K, 0)
        control = np.exp(-r*T) * S_T

        beta = np.cov(discounted_payoff, control)[0,1] / np.var(control)

        cv_estimator = discounted_payoff + beta * (np.mean(control) - control)
        cv_price = np.mean(cv_estimator)
    else:
        cv_price = None

    # ============================
    # CONFIDENCE INTERVAL
    # ============================
    discounted = np.exp(-r*T) * payoffs
    std_error = np.std(discounted) / np.sqrt(len(discounted))
    lower = mc_price - 1.96 * std_error
    upper = mc_price + 1.96 * std_error

    # ============================
    # GREEKS
    # ============================
    delta, gamma, vega, theta = compute_greeks(S0, K, r, sigma, T)

    # ============================
    # DASHBOARD
    # ============================
    st.subheader("📊 Pricing Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Monte Carlo Price", f"{mc_price:.4f}")
    col2.metric("Black-Scholes Price", f"{bs_price:.4f}")

    if cv_price:
        col3.metric("Control Variate Price", f"{cv_price:.4f}")

    st.write(f"**95% Confidence Interval:** ({lower:.4f}, {upper:.4f})")
    st.write(f"**Execution Time:** {end - start:.4f} sec")

    # ============================
    # PLOTS
    # ============================

    # 1. GBM Paths
    st.subheader("📈 GBM Simulation")
    fig, ax = plt.subplots()
    ax.plot(paths[:, :20])
    st.pyplot(fig)

    # 2. Payoff Distribution
    st.subheader("📊 Payoff Distribution")
    fig, ax = plt.subplots()
    ax.hist(discounted, bins=50)
    st.pyplot(fig)

    # 3. Convergence
    st.subheader("📉 Convergence")
    running_avg = np.cumsum(discounted) / np.arange(1, len(discounted)+1)

    fig, ax = plt.subplots()
    ax.plot(running_avg)
    ax.axhline(bs_price, linestyle='--')
    st.pyplot(fig)

    # 4. Greeks
    st.subheader("📊 Greeks")
    st.write({
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta
    })

    # 5. Volatility Smile
    st.subheader("📈 Volatility Smile")

    strike_range = np.linspace(80, 120, 10)
    ivs = []

    for K_i in strike_range:
        price = black_scholes_call(S0, K_i, r, sigma, T)
        ivs.append(sigma)

    fig, ax = plt.subplots()
    ax.plot(strike_range, ivs, marker='o')
    st.pyplot(fig)
