__author__ = 'Shawn Li'

import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import heapq

def get_log_return(data):
    start = data['close'].index[0]
    returns = [data['close'][i+1+start]/data['close'][i+start] for i in range(0,data['close'].__len__()-1)]
    log_return = np.log(returns)
    log_return = [i for i in log_return]
    return log_return

def normalization(weight_list):
    weight_list = np.array(weight_list)
    weight_list = weight_list/sum(weight_list)
    weight_list = [i for i in weight_list]
    return weight_list


code_list = [i for i in ts.get_hs300s()['code']]
weight_list = [i for i in ts.get_hs300s()['weight']]
data = [0  for i in range(code_list.__len__())]


for i in range(code_list.__len__()):
    temp_data = ts.get_k_data(code = code_list[i],start='2014-06-01',end='2017-06-01', ktype = 'D')
    data[i] =  len(temp_data)

date_num = max(data)
new_code_list = []
new_weight_list = []
new_data = []

for i in range(code_list.__len__()):
    data = ts.get_k_data(code = code_list[i],start='2014-06-01',end='2017-06-01', ktype = 'D')
    if len(data) == date_num:
        new_data.append(get_log_return(data))
        new_code_list.append(code_list[i])
        new_weight_list.append(weight_list[i])

stock_num = len(new_weight_list)
top_num = int(stock_num*0.05)
portfolio_weight = [0 for i in range(date_num-2)]

portfolio_return1 = [0 for i in range(date_num-2)]
portfolio_return2 = [0 for i in range(date_num-2)]

for i in range(date_num-2):
    return_list1 = [0 for i in range(stock_num)]
    return_list2 = [0 for i in range(stock_num)]
    for j in range(stock_num):
        return_list1[j] = new_data[j][i]
        return_list2[j] = new_data[j][i+1]
    return_list1 = np.array(return_list1)

    index1 = heapq.nlargest(top_num, range(stock_num), return_list1.take)
    index2 = heapq.nsmallest(top_num, range(stock_num), return_list1.take)

    day_weight1 = [0 for i in range(top_num)]
    day_return1 = [0 for i in range(top_num)]

    day_weight2 = [0 for i in range(top_num)]
    day_return2 = [0 for i in range(top_num)]

    for k in range(top_num):
        day_weight1[k] = new_weight_list[index1[k]]
        day_return1[k] = return_list2[index1[k]]
    day_weight1 = normalization(day_weight1)

    day_return1 = np.array(day_return1)
    day_weight1 = np.array(day_weight1)
    returns1 = np.dot(day_return1,day_weight1.T)
    portfolio_return1[i] = returns1

    for k in range(top_num):
        day_weight2[k] = new_weight_list[index2[k]]
        day_return2[k] = return_list2[index2[k]]
    day_weight2 = normalization(day_weight2)

    day_return2 = np.array(day_return2)
    day_weight2 = np.array(day_weight2)
    returns2 = np.dot(day_return2,day_weight2.T)
    portfolio_return2[i] = returns2

def get_value(returns):
    length = len(returns)
    portfolio_value = [0 for i in range(length+1)]
    portfolio_value[0] = 1
    for i in range(length):
        portfolio_value[i+1] = portfolio_value[i]+returns[i]
    return portfolio_value

index_return = get_log_return(ts.get_k_data(code = '000300', index = True, start='2014-06-01',end='2017-06-01', ktype = 'D'))[1:]
index_value = get_value(index_return)
index_value = np.array(index_value)

portfolio_value1 = get_value(portfolio_return1)
portfolio_value1 = np.array(portfolio_value1)
portfolio_value2 = get_value(portfolio_return2)
portfolio_value2 = np.array(portfolio_value2)
sns.plt.plot(portfolio_value1,label = 'momentum')
sns.plt.plot(portfolio_value2,label = 'reversal')
sns.plt.plot(index_value,label = 'index')
sns.plt.legend()
sns.plt.show()







