"""Monte Carlo simulation for European Option Pricing"""

import reference_data as reference_data

import numpy as np

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
    def __init__(
        self, initial_price: float, volatility, dt: float = 1/365, t_ex: float = reference_data.maturity):
        self.current_price = initial_price
        self.initial_price = initial_price
        self.volatility = volatility
        self.dt = dt
        self.t_ex = t_ex
        self.prices = []
        self.simulate_paths()

    def simulate_paths(self):
        """
        Method to simulate path over the horizon t_ex
        """
        while(self.t_ex - self.dt > 0):
            dWt = np.random.normal(0, np.sqrt(self.dt))  # Brownian motion
            dYt = self.volatility*dWt  # Change in price
            self.current_price += dYt  # Add the change to the current price
            self.prices.append(self.current_price)  # Append new price to series
            self.t_ex -= self.dt  # Accound for the step in time