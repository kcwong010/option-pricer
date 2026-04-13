import numpy as np
from scipy.stats import norm , qmc
import math
import pandas as pd

class OptionPricer():

    def __init__(self, S0, K, T , r, sigma, t = 0, q = 0):
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
    

    def get_american_option_price(self, n_steps=100, option_type='call'):
        """
        Calculate the price of an American option using a binomial tree model.

        Args:
            n_steps (int): Number of steps in the binomial tree
            option_type (str): 'call' for call option, 'put' for put option
        Returns:
            float: The price of the American option
        """
        # Use the passed n_steps, not self.n_steps
        dT = self.tau / n_steps
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1.0 / u
        discount_factor = np.exp(-self.r * dT)
        p = (np.exp((self.r - self.q) * dT) - d) / (u - d)
        
        if not (0.0 <= p <= 1.0):
            raise ValueError("Risk-neutral probability out of bounds. Decrease step size or check rates.")

        # Initialize the stock prices at maturity (Step N)
        # S_at_terminal[i] = S0 * u^(n_steps - i) * d^i
        i_idx = np.arange(n_steps + 1)
        S_nodes = self.S0 * (u ** (n_steps - i_idx)) * (d ** i_idx)

        # Initialize option values at maturity
        if option_type == 'call':
            V = np.maximum(S_nodes - self.K, 0.0)
        else:
            V = np.maximum(self.K - S_nodes, 0.0)

        # Backward induction
        for j in range(n_steps - 1, -1, -1):
            # Calculate continuation values
            # V[i] is updated using V[i] (up) and V[i+1] (down) from the previous (j+1) step
            V = discount_factor * (p * V[:-1] + (1.0 - p) * V[1:])
            
            # Calculate current stock prices at this step j to check for early exercise
            i_idx = np.arange(j + 1)
            S_current = self.S0 * (u ** (j - i_idx)) * (d ** i_idx)
            
            # American feature: Exercise vs Continuation
            if option_type == 'call':
                exercise = np.maximum(S_current - self.K, 0.0)
            else:
                exercise = np.maximum(self.K - S_current, 0.0)
                
            V = np.maximum(V, exercise)

        return float(V[0])
    

    def get_geometric_asian_option_price(self, n_obs, option_type='call'):

        #Adjusted Volatility
        sigma_adj = self.sigma * np.sqrt((n_obs + 1)*(2 * n_obs + 1)/(6 * n_obs**2))

        #Adjusted Cost-of-Carry
        b_adj = ((n_obs+1)/(2 * n_obs)) * (self.r - self.q - 0.5 * self.sigma**2) + 0.5 * sigma_adj**2

        d1_adj = (np.log(self.S0 / self.K) + (b_adj + 0.5 * sigma_adj**2) * self.tau) / (sigma_adj * np.sqrt(self.tau))
        d2_adj = d1_adj - sigma_adj * np.sqrt(self.tau)

        if option_type == 'call':
            price = np.exp((b_adj-self.r) * self.tau) * self.S0 * norm.cdf(d1_adj) - self.K * np.exp(-self.r * self.tau) * norm.cdf(d2_adj)
        else:
            price = self.K * np.exp(-self.r * self.tau) * norm.cdf(-d2_adj) - self.S0 * np.exp((b_adj-self.r) * self.tau) * norm.cdf(-d1_adj)

        return price


    def get_arithmetic_asian_option_price(self, n_obs, option_type='call', num_simulations=100000):

        dt = self.tau / n_obs

        # 1. Generate paths for Arithmetic and Geometric
        # Using log-normal growth: S_t = S_{t-1} * exp((r - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z)
        np.random.seed(42)
        Z = np.random.standard_normal((num_simulations, n_obs))
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        # Calculate price paths
        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=1)
        paths = self.S0 * np.exp(cum_log_returns)

        # 2. Calculate Averages
        arith_avg = np.mean(paths, axis=1)
        # Geometric mean: exp(1/n * sum(log(S_i)))
        geom_avg = np.exp(np.mean(np.log(paths), axis=1))

        # 3. Calculate Payoffs
        if option_type == 'call':
            payoff_arith = np.maximum(arith_avg - self.K, 0) * np.exp(-self.r * self.tau)
            payoff_geom = np.maximum(geom_avg - self.K, 0) * np.exp(-self.r * self.tau)
        else:
            payoff_arith = np.maximum(self.K - arith_avg, 0) * np.exp(-self.r * self.tau)
            payoff_geom = np.maximum(self.K - geom_avg, 0) * np.exp(-self.r * self.tau)

        # Calculate true geometric Asian option price using closed-form solution
        exact_geom = self.get_geometric_asian_option_price(n_obs=n_obs, option_type=option_type)

        # 5. Control Variate Adjustment
        # Calculate Beta (regression coefficient)
        cov_matrix = np.cov(payoff_arith, payoff_geom)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # Adjusted Arithmetic Price
        payoff_cv = payoff_arith - beta * (payoff_geom - exact_geom)
        
        price_mc = np.mean(payoff_arith)
        price_cv = np.mean(payoff_cv)
        std_mc = np.std(payoff_arith) / np.sqrt(num_simulations)
        std_cv = np.std(payoff_cv) / np.sqrt(num_simulations)

        return price_mc, price_cv, std_mc, std_cv
    

    def get_geometric_basket_option_price(self, S2, sigma2, q2=0, rho=0, option_type='call'):
        """
        Calculate the price of a geometric basket option on two underlying assets.

        Args:
            S2 (float): Price of the second underlying asset.
            sigma2 (float): Volatility of the second underlying asset.
            q2 (float): Dividend yield of the second underlying asset.
            rho (float): Correlation between the two underlying assets.
            option_type (str): Type of the option ('call' or 'put').
        Returns:
            float: The price of the geometric basket option.
        """

        basket_geom_0 = (self.S0 * S2) ** 0.5

        # calculate the sigma and mu for the geometric basket
        sum_cov = self.sigma ** 2 + sigma2 ** 2 + 2 * rho * self.sigma * sigma2
        sigma_bg = np.sqrt(sum_cov) / 2

        avg_sigma_square = (self.sigma ** 2 + sigma2 ** 2) / 2
        mu_bg = self.r - (self.q + q2) / 2 - 0.5 * avg_sigma_square + 0.5 * sigma_bg ** 2

        # calculate d1 and d2 for the geometric basket
        sqrt_tau = np.sqrt(self.tau)
        d1 = (np.log(basket_geom_0 / self.K) + (mu_bg + 0.5 * sigma_bg ** 2) * self.tau) / (sigma_bg * sqrt_tau)
        d2 = d1 - sigma_bg * sqrt_tau

        # pricing formula
        discount_factor = np.exp(-self.r * self.tau)
        expected_val = basket_geom_0 * np.exp(mu_bg * self.tau)

        if option_type == 'call':
            price = discount_factor * (expected_val * norm.cdf(d1) - self.K * norm.cdf(d2))
        else:
            price = discount_factor * (self.K * norm.cdf(-d2) - expected_val * norm.cdf(-d1))

        return price
    

    def get_arithmetic_basket_option_price(self, S2, sigma2, q2=0, rho=0, option_type='call', method="control_variate", confidence_level=0.95, num_simulations=100000):
        """
        Calculate the price of an arithmetic basket option on two underlying assets using Monte Carlo simulation / Control Variate method.

        Args:
            S2 (float): Price of the second underlying asset.
            sigma2 (float): Volatility of the second underlying asset.
            q2 (float): Dividend yield of the second underlying asset.
            rho (float): Correlation between the two underlying assets.
            option_type (str): Type of the option ('call' or 'put').
            method (str): Method to use for pricing ('monte_carlo' or 'control_variate').
            confidence_level (float): Confidence level for the confidence interval.
            num_simulations (int): Number of Monte Carlo simulations to run.
        Returns:
            float: The price of the arithmetic basket option.
        """

        np.random.seed(42)

        # correlate normals
        Z1 = np.random.normal(size=num_simulations)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=num_simulations)

        # simulate terminal prices
        sqrt_tau = np.sqrt(self.tau)
        drift1 = (self.r - self.q - 0.5 * self.sigma**2) * self.tau
        drift2 = (self.r - q2 - 0.5 * sigma2**2) * self.tau
        S1_T = self.S0 * np.exp(drift1 + self.sigma * sqrt_tau * Z1)
        S2_T = S2 * np.exp(drift2 + sigma2 * sqrt_tau * Z2)

        # calculate payoffs
        basket_arith = (S1_T + S2_T) / 2
        basket_geom = np.sqrt(S1_T * S2_T)
        discount_factor = np.exp(-self.r * self.tau)

        if option_type == "call":
            payoff_arith = discount_factor * np.maximum(basket_arith - self.K, 0)
            payoff_geom = discount_factor * np.maximum(basket_geom - self.K, 0)
        else:
            payoff_arith = discount_factor * np.maximum(self.K - basket_arith, 0)
            payoff_geom = discount_factor * np.maximum(self.K - basket_geom, 0)

        if method == "monte_carlo":
            price = np.mean(payoff_arith)
            std_error = np.std(payoff_arith) / np.sqrt(num_simulations)
            z_score = norm.ppf(confidence_level)
            ci_lower = price - z_score * std_error
            ci_upper = price + z_score * std_error

            return price, std_error, ci_lower, ci_upper
        
        elif method == "control_variate":
            # calculate the price of the geometric basket option using the closed-form solution
            price_geom = self.get_geometric_basket_option_price(S2=S2, sigma2=sigma2, q2=q2, rho=rho, option_type=option_type)

            # calculate beta for control variate
            cov_matrix = np.cov(payoff_arith, payoff_geom)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]

            # adjust arithmetic payoff using control variate
            payoff_cv = payoff_arith - beta * (payoff_geom - price_geom)

            price_cv = np.mean(payoff_cv)
            std_error_cv = np.std(payoff_cv) / np.sqrt(num_simulations)
            z_score = norm.ppf(confidence_level)
            ci_lower_cv = price_cv - z_score * std_error_cv
            ci_upper_cv = price_cv + z_score * std_error_cv

            return price_cv, std_error_cv, ci_lower_cv, ci_upper_cv


    def get_KIKO_put_option_price(self, barrier_lower, barrier_upper, Rebate, confidence_level=0.95, N=24):

        # delta t
        deltaT = self.tau/N

        # set the random seed
        seed = 42
        np.random.seed(seed)

        # generate the paths of stock prices
        values = []
        M = int(1e2)
        sequencer = qmc.Sobol(d=N, seed=seed)
        # uniform samples
        X = np.array(sequencer.random(n=M))
        # standard normal samples
        Z = norm.ppf(X)
        # scaled samples
        samples = (self.r - self.q - 0.5 * self.sigma ** 2) * deltaT + self.sigma * math.sqrt(deltaT) * Z
        df_samples = pd.DataFrame(samples)
        df_samples_cumsum = df_samples.cumsum(axis=1)
        # the simulated stock prices
        df_stocks = self.S0 * np.exp(df_samples_cumsum)
        for ipath in df_stocks.index.to_list():
            ds_path_local = df_stocks.loc[ipath, :]
            price_max = ds_path_local.max()
            price_min = ds_path_local.min()

            if price_max >= barrier_upper:    # (1) knock-out happened
                knockout_time = ds_path_local[ds_path_local >= barrier_upper].index.to_list()[0]
                payoff = Rebate * np.exp(-knockout_time * self.r * deltaT)
                values.append(payoff)
            elif price_min <= barrier_lower:# (2) knock-in happend
                final_price = ds_path_local.iloc[-1]
                payoff = np.exp(- self.r * self.tau) * max(self.K - final_price, 0)
                values.append(payoff)
            else: # (3) no knock-in, no knock-out, payoff is zero
                values.append(0)

        value = np.mean(values)
        std = np.std(values)
        value_std = std / math.sqrt(M)
        z_score = norm.ppf(confidence_level)
        conf_interval_lower = value - z_score * value_std
        conf_interval_upper = value + z_score * value_std

        return value, conf_interval_lower,conf_interval_upper, value_std
