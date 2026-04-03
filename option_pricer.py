import numpy as np
from scipy.stats import norm

class OptionPricer():

    def __init__(self, S0, K, T , r,  sigma,t = 0,q = 0):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula
        
        Parameters:
        -----------
        S0 : float
            Current asset price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        t : float
            Current time (usually 0)
        r : float
            Risk-free interest rate (as decimal)
        sigma : float
            Volatility (as decimal)
        q : float
            Dividend yield (as decimal, default 0)
        
        Returns:
        --------
        tuple
            (d1, d2) values
        """
        self.S0 = S0
        self.K = K
        self.t = t
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.tau = T - t
        if self.tau <= 0:
            raise ValueError("Time to maturity must be positive")

    def calculate_d1_d2(self):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula
        
        --------
        tuple
            (d1, d2) values
        """
        
        # Calculate d1 and d2
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        d2 = d1 - self.sigma * np.sqrt(self.tau)

        return d1, d2
    
    def get_european_option_price(self, option_type='call'):
        """
        Calculates the price of a European call or put option using the Black-Scholes formula.
        Args:
            option_type (str): 'call' for call option, 'put' for put option.
        Returns:
            float: The price of the option.
        """
        d1, d2 = self.calculate_d1_d2()
        if option_type == 'call':
            return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.tau) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

    # def get_implied_volatility(self, market_price, option_type='call'):
    #     """
    #     Calculate implied volatility using the Black-Scholes model
        
    #     Parameters:
    #     -----------

    #     market_price : float
    #         Market price of the option
    #     option_type : str
    #         'call' or 'put'

    #     Returns:
    #     --------
    #     float
    #         Implied volatility (as decimal)
    #     """
    #     # Initial guess for volatility

    #     sigmahat = np.sqrt(2*abs( (np.log(self.S0/self.K) + (self.r-self.q)*self.tau)/self.tau ) )
    #     tol = 1e-8
    #     nmax = 500
    #     sigmadiff = 1
    #     n=1
    #     sigma = sigmahat
        

    #     while (sigmadiff >= tol and n < nmax):
    #         if option_type == 'call':
    #             price = self.get_european_option_price(option_type='call')
    #         else:
    #             price = self.get_european_option_price(option_type='put')

    #         vega = self.S0 * np.exp(-self.q * self.tau) * norm.pdf((np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.tau) / (sigma * np.sqrt(self.tau))) * np.sqrt(self.tau)
    #         if vega == 0:
    #             return float('NaN')
    #         increment = (price - market_price) / vega
    #         sigma = sigma - increment
    #         n = n+1
    #         sigmadiff = abs(increment)
    #     return sigma
    
    def get_american_option_price(self, n_steps=100, option_type='call'):
        """
        Calculates the price of an American call or put option using a binomial tree model.
        Args:
            n_steps (int): The number of steps in the binomial tree.
            option_type (str): 'call' for call option, 'put' for put option.
        Returns:
            float: The price of the option.
        """

        # inputs: n_steps, sigma, T, S0, K, option_type, r, q
        dT = self.tau / self.n_steps
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1.0 / u
        discount_factor = np.exp(-self.r * dT)
        p = (np.exp((self.r - self.q) * dT) - d) / (u - d)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Risk-neutral probability out of bounds; check inputs or reduce step size")

        # price tree
        S = np.full((n_steps + 1, n_steps + 1), np.nan, dtype=float)
        S[0, 0] = self.S0
        for j in range(1, n_steps + 1):
            i = np.arange(j + 1)
            S[: j + 1, j] = self.S0 * (u ** (j - i)) * (d ** i)

        # payoff tree (full matrix)
        V = np.full_like(S, np.nan)
        if option_type == 'call':
            V[:, n_steps] = np.maximum(S[:, n_steps] - self.K, 0.0)
        else:
            V[:, n_steps] = np.maximum(self.K - S[:, n_steps], 0.0)

        # backward induction (column-by-column)
        for j in range(n_steps - 1, -1, -1):
            cont = discount_factor * (p * V[: j + 1, j + 1] + (1.0 - p) * V[1 : j + 2, j + 1])
            if option_type == 'call':
                exercise = np.maximum(S[: j + 1, j] - self.K, 0.0)
            else:
                exercise = np.maximum(self.K - S[: j + 1, j], 0.0)
            V[: j + 1, j] = np.maximum(cont, exercise)

        return float(V[0, 0])
    
    def get_geometric_asian_option_price(self, option_type='call'):
        return None
    
    def get_arithmetic_asian_option_price(self, option_type='call', num_simulations=10000):
        return None
    
    def get_geometric_basket_option_price(self, option_type='call'):
        return None
    
    def get_arithmetic_basket_option_price(self, option_type='call', num_simulations=10000):
        return None
    
    def get_KIKO_put_option_price(self, option_type='put'):
        return None