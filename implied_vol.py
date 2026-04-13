import numpy as np
from scipy.stats import norm
from option_pricer import get_european_option_price


def get_implied_volatility(S0, K, T, r, q, market_price, option_type='call'):
    """
    Calculate implied volatility using the Black-Scholes model

    Args:
        S0 (float): Current asset price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        q (float): Dividend yield
        market_price (float): observed market price of the option
        option_type (str): 'call' for call option, 'put' for put option
    Returns:
        float: Implied volatility (as decimal)
    """
    # Initial guess for volatility

    sigmahat = np.sqrt(2*abs( (np.log(S0/K) + (r-q)*T)/T ) )
    tol = 1e-8
    nmax = 500
    sigmadiff = 1
    n=1
    sigma = sigmahat
    
    while (sigmadiff >= tol and n < nmax):
        if option_type == 'call':
            price = get_european_option_price(S0=S0, K=K, T=T, r=r, sigma=sigma, t=0, q=q, option_type='call')
        else:
            price = get_european_option_price(S0=S0, K=K, T=T, r=r, sigma=sigma, t=0, q=q, option_type='put')

        vega = S0 * np.exp(-q * T) * norm.pdf((np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return float('NaN')
        increment = (price - market_price) / vega
        sigma = sigma - increment
        n = n+1
        sigmadiff = abs(increment)

    return sigma
