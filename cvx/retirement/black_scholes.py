import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, r, T, sigma, option_type):
    """
    param S: stock price
    param K: strike price
    param r: risk-free rate
    param T: time to maturity
    param sigma: volatility
    param option_type: 'call' or 'put'

    returns: option price based on the Black-Scholes model
    """

    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type == 'put':
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

def find_call_strike(cost, S, r, T, sigma):
    """
    param cost: cost of the call option
    param S: stock price
    param r: risk-free rate
    param T: time to maturity
    param sigma: volatility

    returns: strike price of the call option based on the Black-Scholes model
    """

    U = 10 * S
    L = 0
    tol = 1e-6

    while U - L > tol:
        mid = (U + L) / 2

        if black_scholes_price(S, mid, r, T, sigma, 'call') > cost:
            L = mid
        else:
            U = mid

    return L

def get_collar(r, T, sigma, downside):
    """
    param S: stock price
    param r: risk-free rate
    param T: time to maturity
    param sigma: volatility
    param cost: cost of the call option
    param downside: percentage of downside

    returns: upside and downside caps of the collar
    """
    S = 1
    K_put = S * (1 + downside)

    cost_put = black_scholes_price(S, K_put, r, T, sigma, 'put')
    # print(cost_put)
    K_call = find_call_strike(cost_put, S, r, T, sigma)

    upside = K_call / S - 1

    return downside, upside