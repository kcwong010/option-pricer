import streamlit as st
import math
from option_pricer import OptionPricer
from implied_vol import get_implied_volatility

st.set_page_config(page_title="OptionPricer", layout="wide")
st.title("OptionPricer — Products")

product = st.selectbox("Product", [
    "European", "American", "Geometric Asian", "Arithmetic Asian",
    "Geometric Basket", "Arithmetic Basket", "KIKO Put", "Implied Volatility"
])

# Common inputs
col1, col2 = st.columns(2)
with col1:
    S0 = st.number_input("S0", value=100.0)
    K = st.number_input("K (strike)", value=100.0)
    T = st.number_input("T (years)", value=1.0)
    r = st.number_input("r (risk-free)", value=0.05)
    q = st.number_input("q (dividend)", value=0.0)
with col2:
    option_type = st.selectbox("Option type", ["call", "put"])
    
    if product != "Implied Volatility":  # Implied vol doesn't take sigma as input
        sigma = st.number_input("sigma (asset1)", value=0.2)
        
    # Only show MC controls for products that use simulation
    if product in ("Arithmetic Asian", "Arithmetic Basket"):
        method = st.selectbox("MC method", ["monte_carlo", "control_variate"])
        num_simulations = int(st.number_input("Simulations", value=100000, step=1000))
    else:
        method = "n/a"
        num_simulations = 0

# Confidence level default (used for KIKO and Arithmetic Basket)
confidence_level = 0.95

# Product-specific inputs
if product in ("Geometric Basket", "Arithmetic Basket"):
    S2 = st.number_input("S2", value=100.0)
    sigma2 = st.number_input("sigma2 (asset2)", value=0.2)
    q2 = st.number_input("q2 (dividend for asset2)", value=0.0)
    rho = st.slider("rho (correlation)", -1.0, 1.0, 0.5)
    if product == "Arithmetic Basket":
        confidence_level = float(st.number_input("Confidence level", value=0.95, min_value=0.5, max_value=0.999, step=0.01))
if product in ("Geometric Asian", "Arithmetic Asian"):
    n_obs = int(st.number_input("Observations (n_obs)", value=12, step=1))
if product == "American":
    n_steps = int(st.number_input("Binomial steps (n_steps)", value=100, step=1))
if product == "KIKO Put":
    barrier_lower = st.number_input("Barrier lower", value=80.0)
    barrier_upper = st.number_input("Barrier upper", value=120.0)
    Rebate = st.number_input("Rebate", value=5.0)
    N = int(st.number_input("Path steps (N)", value=24, step=1))
    confidence_level = float(st.number_input("Confidence level", value=0.95, min_value=0.5, max_value=0.999, step=0.01))

if product == "Implied Volatility":
    market_price = float(st.number_input("Market price", value=5.0))

# Price action
if st.button("Price"):
    try:
        if product == "Implied Volatility":
            # For implied vol, we don't need to create an OptionPricer instance
            implied_vol = get_implied_volatility(S0=S0, K=K, T=T, r=r, q=q, market_price=market_price, option_type=option_type)
            if math.isnan(implied_vol):
                st.error("Implied volatility did not converge")
            else:
                st.metric("Implied volatility", f"{implied_vol:.6f}")
            st.stop()  # Skip the rest of the pricing logic for implied vol
        else:
            pricer = OptionPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, t=0.0, q=q)
            if product == "European":
                val = pricer.get_european_option_price(option_type=option_type)
                st.metric("European price", f"{val:.6f}")
            elif product == "American":
                val = pricer.get_american_option_price(n_steps=n_steps, option_type=option_type)
                st.metric("American price", f"{val:.6f}")
            elif product == "Geometric Asian":
                val = pricer.get_geometric_asian_option_price(n_obs=n_obs, option_type=option_type)
                st.metric("Geometric Asian price", f"{val:.6f}")
            elif product == "Arithmetic Asian":
                price_mc, price_cv, se_mc, se_cv = pricer.get_arithmetic_asian_option_price(n_obs=n_obs, option_type=option_type, num_simulations=num_simulations)
                if method == "monte_carlo":
                    st.metric("Arithmetic Asian (MC)", f"{price_mc:.6f}")
                    st.write("Std. error:", f"{se_mc:.6f}")
                    st.write("95% CI:", f"[{price_mc - 1.96 * se_mc:.6f}, {price_mc + 1.96 * se_mc:.6f}]")
                elif method == "control_variate":
                    st.metric("Arithmetic Asian (CV)", f"{price_cv:.6f}")
                    st.write("Std. error (CV):", f"{se_cv:.6f}")
                    st.write("95% CI:", f"[{price_cv - 1.96 * se_cv:.6f}, {price_cv + 1.96 * se_cv:.6f}]")
            elif product == "Geometric Basket":
                val = pricer.get_geometric_basket_option_price(S2=S2, sigma2=sigma2, q2=q2, rho=rho, option_type=option_type)
                st.metric("Geometric Basket price", f"{val:.6f}")
            elif product == "Arithmetic Basket":
                if method == "monte_carlo":
                    price, se, ci_low, ci_high = pricer.get_arithmetic_basket_option_price(S2=S2, sigma2=sigma2, q2=q2, rho=rho, option_type=option_type, method="monte_carlo", num_simulations=num_simulations, confidence_level=confidence_level)
                    st.metric("Arithmetic Basket (MC)", f"{price:.6f}")
                    st.write("Std. error:", f"{se:.6f}")
                    st.write("95% CI:", f"[{ci_low:.6f}, {ci_high:.6f}]")
                else:
                    price_cv, se_cv, ci_low_cv, ci_high_cv = pricer.get_arithmetic_basket_option_price(S2=S2, sigma2=sigma2, q2=q2, rho=rho, option_type=option_type, method="control_variate", num_simulations=num_simulations, confidence_level=confidence_level)
                    st.metric("Arithmetic Basket (CV)", f"{price_cv:.6f}")
                    st.write("Std. error (CV):", f"{se_cv:.6f}")
                    st.write("95% CI (CV):", f"[{ci_low_cv:.6f}, {ci_high_cv:.6f}]")
            elif product == "KIKO Put":
                value, conf_interval_lower,conf_interval_upper, value_std = pricer.get_KIKO_put_option_price(barrier_lower, barrier_upper, Rebate, confidence_level=confidence_level, N=N)
                st.metric("KIKO Put price", f"{value:.6f}")
                st.write("Std. error:", f"{value_std:.6f}")
                st.write("95% CI:", f"[{conf_interval_lower:.6f}, {conf_interval_upper:.6f}]")
            elif product == "Implied Volatility":
                implied_vol = get_implied_volatility(S0=S0, K=K, T=T, r=r, q=q, market_price=market_price, option_type=option_type)
                if math.isnan(implied_vol):
                    st.error("Implied volatility did not converge")
                else:
                    st.metric("Implied volatility", f"{implied_vol:.6f}")
    except Exception as e:
        st.error(f"Pricing error: {e}")