__author__ = 'Shawn Li'
#-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import arch
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def fit_garch_model(file):
    data = pd.read_excel(file).dropna()
    data.drop(0,inplace=True)
    returns = (np.array(data[u'涨跌幅%']))/100+1
    returns = np.array(returns,dtype=np.float64)
    log_return = np.log(returns)
    garch_model = arch_model(log_return,p=1,q=1,dist='StudentsT')
    fit_res = garch_model.fit()
    param = fit_res.params
    volatility = fit_res.conditional_volatility
    return param,volatility

def calculate_culmative_yield(data):
    culmative_yield_list = [0 for i in range(0,len(data))]
    last_close = data[u'收盘价'][0]
    for i in range(0,len(data)):
        if '15:00:00' in str(data[u'交易时间'][i]):
            close=data[u'收盘价'][i]
            culmative_yield_list[i] = (close-last_close)/last_close*100
            last_close=data[u'收盘价'][i]

        else:
            close=data[u'收盘价'][i]
            culmative_yield_list[i] = (close-last_close)/last_close*100
    culmative_yield_list[0] = data[u'涨跌幅%'][0]
    data[u'累计收益率'] = culmative_yield_list
    return data

def up_find(data,per,flag_list):
    per_list = [i for i in data[data[u'累计收益率']>per].index]
    for i in per_list:
        if i == 0:
            continue
        if ((data[u'累计收益率'][i]-per)*(data[u'累计收益率'][i-1]-per))<0:
            flag_list[i] = 1

def down_find(data,per,flag_list):
    per_list = [i for i in data[data[u'累计收益率']<per].index]
    for i in per_list:
        if i== 1:
            continue
        if ((data[u'累计收益率'][i]-per)*(data[u'累计收益率'][i-1]-per))<0:
            flag_list[i] = -1

def update_flag(data):
    flag_list = [0 for i in range(0,len(data))]
    for i in range(1,11):
        up_find(data,i,flag_list)
    for i in range(-10,0):
        down_find(data,i,flag_list)
    for i in range(1,len(data)+1):
        if '09:35:00' in str(data[u'交易时间'][i]):
            flag_list[i] = 0
    data[u'是否穿过'] = flag_list
    return data

def data_process(data):
    vt_data=DataFrame(columns=['v1','v2','t1','t2'])
    t1_list=[]
    t2_list=[]
    v1_list=[]
    v2_list=[]
    for i in range(1,len(data)+1):
        if '09:35:00' in str(data[u'交易时间'][i]) or '09:40:00' in str(data[u'交易时间'][i]):
            continue

        if data[u'是否穿过'][i]==0:
            continue
        elif data[u'是否穿过'][i]==1:
            v1_list.append(data[u'波动率'][i])
            v2_list.append(data[u'波动率'][i-1])
            t1_list.append(data[u'成交量'][i])
            t2_list.append(data[u'成交量'][i-1])
        elif data[u'是否穿过'][i]==-1:
            v1_list.append(data[u'波动率'][i])
            v2_list.append(data[u'波动率'][i-1])
            t1_list.append(data[u'成交量'][i])
            t2_list.append(data[u'成交量'][i-1])

    vt_data['v1']=v1_list
    vt_data['v2']=v2_list
    vt_data['t1']=t1_list
    vt_data['t2']=t2_list
    return vt_data

def t_test(data1,data2):
    array1=np.array(data1)
    array2=np.array(data2)
    u=(array1-array2).mean()
    flag=stats.levene(data1,data2)
    if flag[1]<0.05:
        result=stats.ttest_ind(array1,array2, equal_var=False)
    else:
        result=stats.ttest_ind(array1,array2, equal_var=True)

    return result,u

def process(filename):
    data=(pd.read_excel(filename).dropna())
    data = calculate_culmative_yield(data)
    result = fit_garch_model(filename)
    data = data[1:]
    data[u'波动率']=result[1]
    data = update_flag(data)
    res = data_process(data)
    v_ttest_result = t_test(res['v1'],res['v2'])
    t_ttest_result = t_test(res['t1'],res['t2'])
    '''sns.plt.plot(res['v1'])
    sns.plt.plot(res['v2'])
    sns.plt.legend()
    sns.plt.show()
    sns.plt.plot(res['t1'])
    sns.plt.plot(res['t2'])
    sns.plt.legend()
    sns.plt.show()'''
    print(v_ttest_result[0],v_ttest_result[1])
    print(t_ttest_result[0],t_ttest_result[1])
    print(len(res['v1']))

process(u'沪深300数据.xls')
process(u'上证指数数据.xls')
process(u'中小板指数据.xls')
process(u'创业板指数据.xls')




