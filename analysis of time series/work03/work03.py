__author__ = 'Shawn Li'

import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime
import statsmodels.tsa.ar_model as ar
import numpy as np


def get_log_return(data):
    returns = [data['close'][i+1]/data['close'][i] for i in range(0,data['close'].__len__()-1)]
    log_return = np.log(returns)
    log_return = [i for i in log_return]
    return log_return

def fit_ar_model(data):
    log_return = get_log_return(data)
    date = [i for i in data['date']]
    xs = [datetime.strptime(d, '%Y-%m-%d') for d in date]
    obj = Series(log_return, index = xs[1:])
    a = ar.AR(endog = obj)
    fit_model = a.fit(ic = 'aic',  trend = 'c',full_output = 1,disp = 1)
    order = fit_model.k_ar
    param = fit_model.params
    root = fit_model.roots
    variance = fit_model.sigma2
    return order,param,root,variance

def normalization(weight_list):
    weight_list = np.array(weight_list)
    weight_list = weight_list/sum(weight_list)
    weight_list = [i for i in weight_list]
    return weight_list

code_list = [i for i in ts.get_hs300s()['code']]
weight_list = [i for i in ts.get_hs300s()['weight']]
data = [0  for i in range(code_list.__len__())]

for i in range(code_list.__len__()):
    data[i] =  len(ts.get_k_data(code = code_list[i],start='2016-01-01',end='2017-01-01', ktype = 'D'))

num = max(data)
new_code_list = []
new_weight_list = []

for i in range(code_list.__len__()):
    data = ts.get_k_data(code = code_list[i],start='2016-01-01',end='2017-01-01', ktype = 'D')
    if len(data) == num:
        new_code_list.append(code_list[i])
        new_weight_list.append(weight_list[i])
order_list = [0  for i in range(0,new_code_list.__len__())]

for i in range(new_code_list.__len__()):
    data =  ts.get_k_data(code = new_code_list[i],start='2016-01-01',end='2017-01-01', ktype = 'D')
    order = fit_ar_model(data)[0]
    order_list[i] = order

order_set = set(order_list)
classification = [[] for i in range(max(order_set))]

for i in range(order_list.__len__()):
    a = classification[order_list[i]-1]
    classification[order_list[i]-1].append((new_code_list[i],i))


portfolio_weight = [[] for i in range(max(order_set))]
for i in range(max(order_set)):
    for stock in classification[i]:
        portfolio_weight[i].append(new_weight_list[stock[1]])
    portfolio_weight[i] = normalization(portfolio_weight[i])


portfolio_return = [[0] for i in range(portfolio_weight.__len__())]

for i in  range(portfolio_return.__len__()):
    total_return = np.array([0.0 for i in range(num-1)])
    for j in range(classification[i].__len__()):
        stock_data = ts.get_k_data(code = classification[i][j][0],start='2016-01-01',end='2017-01-01', ktype = 'D')
        stock_return = np.array(get_log_return(stock_data))
        total_return = portfolio_weight[i][j]*stock_return + total_return
    total_return = [i for i in total_return]
    portfolio_return[i] = total_return


sh_data = ts.get_k_data(code = '000001', index = True,start='2016-01-01',end='2017-01-01', ktype = 'D')
date = [i for i in sh_data['date']]
xs = [datetime.strptime(d, '%Y-%m-%d') for d in date]
portfolio_return_p = [0 for i in range(portfolio_return.__len__())]

for i in range(portfolio_return.__len__()):
    if len(portfolio_return[i])>0:
        obj = Series(portfolio_return[i], index = xs[1:])
        a = ar.AR(endog = obj)
        fit_model = a.fit(ic = 'aic',  trend = 'c',full_output = 1,disp = 1)
        order = fit_model.k_ar
        portfolio_return_p[i] = order
    else:
        portfolio_return_p[i] = 0

ar_portfolio = []
ar_portfolio_order = []

for i in range(portfolio_return_p.__len__()):
    if portfolio_return_p[i] ==i+1:
        ar_portfolio.append(classification[i])
        ar_portfolio_order.append(i+1)

print(ar_portfolio)
print(num)
print(len(new_code_list))
print(order_set)
print(ar_portfolio_order)





