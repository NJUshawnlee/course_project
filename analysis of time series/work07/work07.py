__author__ = 'Shawn Li'

import tushare as ts
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

def get_log_return(data):
    start = data['close'].index[0]
    returns = [data['close'][i+1+start]/data['close'][i+start] for i in range(0,data['close'].__len__()-1)]
    log_return = np.log(returns)
    log_return = [i for i in log_return]
    return log_return

def garch_test(code, start1 = '2016-03-01', end1 = '2017-06-01', start2 = '2013-09-01', end2 = '2014-12-01' ):

    hist_data1 = ts.get_k_data(code,start = start1,end = end1, index=True, ktype = 'D')
    return1 = get_log_return(hist_data1)
    hist_data2 = ts.get_k_data(code,start = start2,end = end2, index=True, ktype = 'D')
    return2 = get_log_return(hist_data2)
    length = min(len(return1),len(return2))
    return1 = return1[:length]
    return2 = return2[:length]

    garch_1=arch_model(return1,p=1,q=1,dist='StudentsT')
    res1=garch_1.fit()

    garch_2=arch_model(return2,p=1,q=1,dist='StudentsT')
    res2=garch_2.fit()

    res1_volatility = res1.conditional_volatility
    res2_volatility = res2.conditional_volatility
    return res1_volatility,res2_volatility




result = garch_test(code='000001')

sns.plt.plot(result[0],label = 'after')
sns.plt.plot(result[1],label = 'before')
sns.plt.legend()
sns.plt.show()

result = garch_test(code='000300')

sns.plt.plot(result[0],label = 'after')
sns.plt.plot(result[1],label = 'before')

sns.plt.legend()
sns.plt.show()

result = garch_test(code='399005')

sns.plt.plot(result[0],label = 'after')
sns.plt.plot(result[1],label = 'before')

sns.plt.legend()
sns.plt.show()

result = garch_test(code='399006')

sns.plt.plot(result[0],label = 'after')
sns.plt.plot(result[1],label = 'before')

sns.plt.legend()
sns.plt.show()
