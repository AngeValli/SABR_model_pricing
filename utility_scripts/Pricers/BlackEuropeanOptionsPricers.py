"""Pricing of European options with Black formula"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from scipy.stats import norm

import reference_data as reference_data

class BlackEuropeanPricing:
    """
    Closed_form formula for Black-Scholes pricing
    """
    def __init__(
        self, K: float,  implied_volatility: float = reference_data.atm_vol, f: float = reference_data.forward, t_ex: float = reference_data.maturity
            ):
        self.K = K
        self.implied_volatility = implied_volatility
        self.f = f
        self.t_ex = t_ex
        self.d1 = self.Black_d_value(value=1)
        self.d2 = self.Black_d_value(value=2)
        self.call_price = self.Black_call_price()
        self.put_price = self.Black_put_price()
    

    def Black_d_value(self, value: float = 1) -> float:
        """
        Method to compute d1 and d2 values in Black's formula
        Params:
            value (int): value to compute d1 or d2
            forward_price (float): forward_price value, if different from reference_data
         Output:
            d_value (float): Value of d1 or d2
        """
        d_value: float
        log_part: float = np.log(self.f/self.K)
        sigma_part: float = 1/2 * np.power(self.implied_volatility, 2) * self.t_ex
        denominator: float = self.implied_volatility * np.power(self.t_ex, .5)
        if value == 1:
            d_value = (log_part + sigma_part)/denominator
        elif value == 2:
            d_value = (log_part - sigma_part)/denominator
        return d_value

    def Black_call_price(self) -> float:
        """
        Method to compute Black's European option call price
        Output:
            call_price (float): Call price value
        """
        return self.f*norm.cdf(self.d1) - self.K*norm.cdf(self.d2)

    def Black_put_price(self) -> float:
        """
        Method to compute Black's European option put price
        Output:
            put_price (float): Put price value
        """
        return self.call_price + self.K-self.f