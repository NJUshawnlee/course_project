__author__ = 'Shawn Li'

import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime
import statsmodels.tsa.ar_model as ar
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

def get_log_return(data):
    returns = data['close']/data['open']
    returns = [i for i in returns]
    log_return = np.log(returns)
    log_return = [i for i in log_return]
    return log_return

def fit_ar_model(data):
    log_return=get_log_return(data)
    date = [i for i in data['date']]
    xs = [datetime.strptime(d, '%Y-%m-%d') for d in date]
    obj = Series(log_return, index = xs)
    a = ar.AR(endog = obj)
    fit_model = a.fit(ic = 'aic',  trend = 'c',full_output = 1,disp = 1)
    order = fit_model.k_ar
    param = fit_model.params
    root = fit_model.roots
    variance = fit_model.sigma2
    return order,param,root,variance

def n_step_predict(data,params,p,n):
    prediction=[0 for i in range(n)]
    for i in range(0,n):
        prediction[i]=params[0]
        for j in range(1,p+1):
            prediction[i]+=params[j]*data[-j]
        data.append(prediction[i])
    return prediction[n-1]

def out_sample_predict(all_data,period,params,p,n,length):
    prediction = [0 for i in range(period)]
    for i in range(0,period):
        data=all_data[:length+1+i-n]
        prediction[i]=n_step_predict(data,params[i],p,n)
    return prediction

def get_param(data,period,length):
    param_list = [0 for i in range(period)]
    for i in range(period):
        sample_data = data[:length+i-1]
        fit_model = fit_ar_model(sample_data)
        param_list[i] = fit_model[1]
    return param_list

hist_sh_index = ts.get_k_data('000001',start='2016-01-01',end='2017-04-25', index=True, ktype = 'D')
hist_data = get_log_return(hist_sh_index)
hist_fit_ar = fit_ar_model(hist_sh_index)
hist_order = hist_fit_ar[0]
hist_params = hist_fit_ar[1]

pre_sh_index = ts.get_k_data('000001',start='2017-04-25',end='2017-05-12', index=True, ktype = 'D')
pre_data = get_log_return(pre_sh_index)

all_sh_index = ts.get_k_data('000001',start='2016-01-01',end='2017-05-12', index=True, ktype = 'D')
all_data = get_log_return(all_sh_index)

param_list = get_param(all_sh_index, period=len(pre_data), length=len(hist_data))

prediction_1 = out_sample_predict(all_data,period=len(pre_data),params=param_list,p=hist_order,n=1,length=len(hist_data))
prediction_up=prediction_1+(hist_fit_ar[3])**0.5

prediction_down=prediction_1-(hist_fit_ar[3])**0.5
sns.plt.plot(prediction_up,ls=':',color='r')
sns.plt.plot(prediction_down,ls=':',color='r')
sns.plt.plot(prediction_1,marker='o',label='1 step prediction')
sns.plt.plot(pre_data,marker='o',label='reality')
sns.plt.legend()
'''fig = sns.plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test1png.png', dpi=600)'''
sns.plt.show()


