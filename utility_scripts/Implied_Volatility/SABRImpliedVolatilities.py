"""SABR model for implied volatilities approximations"""

import os
import sys

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

import reference_data as reference_data

from typing import List

import numpy as np
import matplotlib.pyplot as plt

class SABR_ATMVol:
    """
    Class for backbone computing and plotting
    """
    def __init__(
        self, f: float = reference_data.forward, alpha: float = reference_data.alpha, beta: float = reference_data.beta, rho: float = reference_data.rho, 
        nu: float = reference_data.vol_of_vol, t_ex: float = reference_data.maturity
              ):
        self.f = f
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.t_ex = t_ex

    def ATM_Implied_Vol(self, forward_price: float = None) -> float:
        """
        Method to compute backbone implied volatility value at-the-money
        Params:
            forward_price (float): forward_price value, if different from reference_data
        Output:
            Backbone value
        """
        if forward_price is None :
            f = self.f
        else:
            f = forward_price
        return self.alpha/np.power(f, 1-self.beta) * (1 +
   (np.power(1-self.beta,2)/24 * np.power(self.alpha,2)/np.power(f, 2-2*self.beta) +
      1/4 * self.rho * self.beta * self.alpha * self.nu/np.power(f, 1-self.beta) +
      (2-3 * np.power(self.rho,2))/24 * np.power(self.nu,2))*self.t_ex)
    
    def display_ATM(self, forward_prices: List[float], figure: bool = False):
        """
        Method to display at-the-money backbone implied volatility
        Params:
            forward_prices (List[float]): forward_prices values to plot the curve
            figure (bool): Boolean value carrying information of existing figure for plot
        """
        curve_implied_vol_ATM: List[str] = list(map(
            lambda forward_price: self.ATM_Implied_Vol(forward_price), forward_prices))
        if not figure:
            plt.figure()
        plt.plot(forward_prices, curve_implied_vol_ATM, linestyle="dashed", label="ATM volatility")
        if not figure:
            plt.title(f"Implied volatility")
            text: str = f"Parameters : alpha={self.alpha}, beta={self.beta}, " +\
                f"rho={self.rho}, nu={self.nu}, t_ex = {self.t_ex}"
            plt.figtext(0.5, -0.01, text, wrap=True, horizontalalignment="center", fontsize=12)
            figure = True
        
        

class SABR_ImpliedVol(SABR_ATMVol):
    def __init__(
        self, K: float, f: float = reference_data.forward, alpha: float = reference_data.alpha, beta: float = reference_data.beta, rho: float = reference_data.rho, 
        nu: float = reference_data.vol_of_vol, t_ex: float = reference_data.maturity
              ):
        super().__init__(f=f, alpha=alpha, beta=beta, rho=rho, nu=nu, t_ex=t_ex)
        self.K = K

    def lambda_ratio(self, forward_price: float = None) -> float:
        """
        Method to compute lambda ratio in SABR model
        """
        if forward_price is None :
            f = self.f
        else:
            f = forward_price
        return self.nu/self.alpha * np.power(f, 1-self.beta)
                                                           
    def Implied_Vol(self, forward_price: float = None) -> float:
        """
        Method to compute implied volatility value
        Params:
            forward_price (float): forward_price value, if different from reference_data
        Output:
            Implied volatility value
        """
        if forward_price is None :
            f = self.f
        else:
            f = forward_price
        lambda_ = self.lambda_ratio()
        return self.alpha/np.power(f, 1-self.beta) * (1 -
        1/2*(1-self.beta-self.rho*lambda_)*np.log(self.K/f) + 1/12 * (np.power(1-self.beta,2) +
          (2-3*np.power(self.rho,2))*np.power(lambda_,2)) * np.power(np.log(self.K/f), 2))
    
    def display_ImpliedVol(self, forward_prices: List[float], figure: bool = False):
        """
        Method to display implied volatility
        Params:
            forward_prices (List[float]): forward_prices values to plot the curve
            figure (bool): Boolean value carrying information of existing figure for plot
        """
        curve_implied_vol: List[str] = list(map(
            lambda forward_price: self.Implied_Vol(forward_price), forward_prices))
        if not figure:
            plt.figure()
        plt.plot(forward_prices, curve_implied_vol, label=f"K={self.K}")
        if not figure:
            plt.title(f"Implied volatility")
            text: str = f"Parameters : alpha={self.alpha}, beta={self.beta}, " +\
                f"rho={self.rho}, nu={self.nu}, t_ex = {self.t_ex}"
            plt.figtext(0.5, -0.01, text, wrap=True, horizontalalignment="center", fontsize=12)
            figure = True