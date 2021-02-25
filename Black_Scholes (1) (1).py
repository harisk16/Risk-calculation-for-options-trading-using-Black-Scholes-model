#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
import matplotlib.pyplot as plt
init_printing()


# In[2]:


def euro_vanilla_call(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call


# In[3]:


euro_vanilla_call(50, 100, 1, 0.05, 0.25)


# In[3]:


def euro_vanilla_put(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put


# In[6]:


euro_vanilla_put(50, 100, 1, 0.05, 0.25)


# In[4]:


def euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = euro_vanilla_call(S, K, T, r, sigma)
    if option == 'put':
        result = euro_vanilla_put(S, K, T, r, sigma)
        
    return result


# In[8]:


import pandas as pd
Data = pd.read_csv('HDFCBANK.csv')
Previous_data = Data['Close'].iloc[4485-124:4485].values
Spot_prices = Data['Close'].iloc[4485:4609].values
Spot_prices_end = Data['Close'].iloc[4609:4609+124].values
str1 = 0.97*Spot_prices 
str2 = 1.12*Spot_prices 
print(str1)
print(len(str1))
St_dev = np.std(Previous_data)
sigma = St_dev/Spot_prices[0]
sigma


# In[9]:


Avg_profit = []
for i in range(len(str1)):
    spi = Spot_prices[i]
    str1i = str1[i]
    str2i = str2[i]
    Final_stock_price = Spot_prices_end[i]
    #On 1st Jan 2019
    Strike_price = np.arange(str1i,str2i)
    Premium = [euro_vanilla(spi, x ,0.5, 0.05, sigma, option = 'call') for x in Strike_price]
    #On 1st April 2019 
    Profit = []
    for j in range(len(Premium)):
        if (Final_stock_price -(Premium[j]+Strike_price[j]) > 0):
            Profit.append(Final_stock_price -(Premium[j]+Strike_price[j]))
        else:
            Profit.append(-Premium[j])
    Avg_profit.append(np.mean(Profit))
    plt.plot(Strike_price, Profit,'r')
    plt.axvline(x=spi)
    plt.ylabel('Profit')
    plt.xlabel('Strike price')
    plt.grid()
    plt.show()
    print(Premium)
plt.plot(Avg_profit)
plt.ylabel('Avg Profit')
#plt.xlabel('Spot price')
plt.grid()
plt.show()
stdev_profit = np.std(Avg_profit)
print(stdev_profit)


# In[10]:


print(Spot_prices)
print(Avg_profit)
Avg_avg_profit = np.mean(Avg_profit)
print(Avg_avg_profit)


# In[20]:


from statistics import NormalDist
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
pdf = NormalDist(mu=Avg_avg_profit, sigma= stdev_profit)
print(1-NormalDist(mu=Avg_avg_profit, sigma= stdev_profit).cdf(0))
x = np.linspace(Avg_avg_profit - 3*stdev_profit, Avg_avg_profit + 3*stdev_profit, 300)
plt.plot(x, stats.norm.pdf(x, Avg_avg_profit, stdev_profit)) 
get_ipython().run_line_magic('plt.axvline', '(x=0)')
plt.show


# In[ ]:




