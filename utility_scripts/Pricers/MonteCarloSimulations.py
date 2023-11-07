"""Monte Carlo simulation for European Option Pricing"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

import reference_data as reference_data

class European_Call_Payoff:
    """
    Class to compute European call payoff
    Params:
        strike (float): Strike value
    """
    def __init__(self, strike: float) -> float:
        self.strike = strike

    def get_payoff(self, stock_price: float) -> float:
        """
        Method to get payoff
        Params:
            stock_price (float) : Stock price value
         Output:
            European call payoff
        """
        if stock_price > self.strike:
            return stock_price - self.strike
        else:
            return 0

class European_Put_Payoff:
    """
    Class to compute European put payoff
    Params:
        strike (float): Strike value
    """
    def __init__(self, strike: float):
        self.strike = strike

    def get_payoff(self, stock_price: float) -> float:
        """
        Method to get payoff
        Params:
            stock_price (float) : Stock price value
         Output:
            European put payoff
        """
        if stock_price < self.strike:
            return self.strike - stock_price 
        else:
            return 0

class GeometricBrownianMotion:
    """
    Class to simulate Monte-Carlo path with Brownian Motion
    Params:
        initial_price (float): Initial stock price value at initial time of the simulation
        current_price (float): Current stock price, initiated at initial value
        volatility (float): Volatility value
        dt (float): Time step
        t_ex (float): Exercise time of the option, maturity
    """
    def __init__(self, initial_price: float, volatility, dt: float = 1/365, t_ex: float = reference_data.maturity, 
                 model: str = "BlackScholes", beta: str = reference_data.beta, alpha: str = reference_data.alpha,
                 nu: str = reference_data.vol_of_vol, rho: str = reference_data.rho):
        self.current_price = initial_price
        self.initial_price = initial_price
        self.volatility = volatility
        self.dt = dt
        self.t_ex = t_ex
        self.prices = []
        self.model = model
        if self.model == "SABR":
            self.beta = beta
            self.alpha = alpha
            self.rho = rho
            self.nu = nu
        self.simulate_paths()

    def simulate_paths(self):
        """
        Method to simulate path over the horizon t_ex
        """
        while(self.t_ex - self.dt > 0):
            if self.model == "BlackScholes":
                dWt = np.random.normal(loc=0, scale=np.sqrt(self.dt))  # Brownian motion
                dYt = self.volatility*dWt  # Change in price
            elif self.model == "SABR":
                dWt_uncorrelated = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=2)  # Brownian motion
                C = np.array([[1, self.rho], [self.rho, 1]]) # Correlation matrix
                L = np.linalg.cholesky(C) # Upper triangular cholesky decomposition
                dWt = np.dot(L, dWt_uncorrelated) # Create correlated brownian motions
                dalpha = self.nu*(self.alpha)**self.beta*dWt[0] # Change in stochastic volatility
                dYt = self.alpha*(self.current_price)**self.beta*dWt[1] # Change in price
                self.alpha += dalpha # Add the change to the current volatility value
            else:
                err_msg: str = "Wrong model parameter or not implemented"
                raise Exception(err_msg)
            self.current_price += dYt  # Add the change to the current price
            self.prices.append(self.current_price)  # Append new price to series
            self.t_ex -= self.dt  # Accound for the step in time