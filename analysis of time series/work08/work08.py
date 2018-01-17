__author__ = 'Shawn Li'
#-*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def get_log_return(data):
    returns = [data[i+1]/data[i] for i in range(0,data.__len__()-1)]
    log_return = np.log(returns)
    log_return = [i for i in log_return]
    return log_return

def normalization(data):
    std = data[0]
    data1 = np. array(data)
    data1 = data1/std
    return data1

def diff(data):
    data = np.array(data)
    length = len(data)
    df = data[1:length]-data[0:length-1]
    return df

def process_fund_data(filename):
    data = (pd.read_excel(filename)).icol(1)
    returns = np.array(get_log_return(data))
    norm_data = normalization(data)
    diff_data =diff(data)
    mu = diff_data.mean()
    mean = returns.mean()
    std = returns.std()
    return norm_data,mu,mean,std

def process_money_data(filename):
    data = np.array((pd.read_excel(filename)).icol(1))
    data = data/100
    diff_data = diff(data)
    mu = diff_data.mean()
    mean = data.mean()
    std = data.std()
    return data,mu,mean,std


"偏股混合型基金"
stock_gongyin = process_fund_data(u'fund/stock/工银瑞信核心价值A.xlsx')
stock_yifangda = process_fund_data(u'fund/stock/易方达科讯.xlsx')
stock_boshi = process_fund_data(u'fund/stock/博时精选A.xlsx')
stock_zhaoshang = process_fund_data(u'fund/stock/招商先锋.xlsx')
stock_nanfang = process_fund_data(u'fund/stock/南方绩优成长A.xlsx')
stock_nuode = process_fund_data(u'fund/stock/诺德价值优势.xlsx')
stock_taixin = process_fund_data(u'fund/stock/泰信先行策略.xlsx')

sns.plt.plot(stock_gongyin[0],label = 'Gongyin',color = 'b')
sns.plt.plot(stock_yifangda[0],label = 'Yifangda',color = 'c')
sns.plt.plot(stock_boshi[0],label = 'Boshi', color = 'g')
sns.plt.plot(stock_zhaoshang[0],label = 'Zhaoshang',color = 'k')
sns.plt.plot(stock_nanfang[0],label = 'Nanfang', color = 'm')
sns.plt.plot(stock_nuode[0],label = 'Nuode',ls=':', color = 'b')
sns.plt.plot(stock_taixin[0],label = 'Taixin',ls=':', color = 'c')
sns.plt.legend()
sns.plt.show()

print(u'工银瑞信核心价值A',stock_gongyin[1],stock_gongyin[2],stock_gongyin[3])
print(u'易方达科讯',stock_yifangda[1],stock_yifangda[2],stock_yifangda[3])
print(u'博时精选A',stock_boshi[1],stock_boshi[2],stock_boshi[3])
print(u'招商先锋',stock_zhaoshang[1],stock_zhaoshang[2],stock_zhaoshang[3])
print(u'南方绩优成长A',stock_nanfang[1],stock_nanfang[2],stock_nanfang[3])
print(u'诺德价值优势',stock_nuode[1],stock_nuode[2],stock_nuode[3])
print(u'泰信先行策略',stock_taixin[1],stock_taixin[2],stock_taixin[3])

"中长期纯债型基金"
bond_gongyin = process_fund_data(u'fund/bond/工银瑞信恒泰纯债.xlsx')
bond_yifangda = process_fund_data(u'fund/bond/易方达富惠.xlsx')
bond_boshi = process_fund_data(u'fund/bond/博时悦楚纯债.xlsx')
bond_zhaoshang = process_fund_data(u'fund/bond/招商招盛A.xlsx')
bond_nanfang = process_fund_data(u'fund/bond/南方多元.xlsx')
bond_zhongjin = process_fund_data(u'fund/bond/中金金利A.xlsx')
bond_tianzhi = process_fund_data(u'fund/bond/天治可转债增强C.xlsx')

print(u'工银瑞信恒泰纯债',bond_gongyin[1],bond_gongyin[2],bond_gongyin[3])
print(u'易方达富惠',bond_yifangda[1],bond_yifangda[2],bond_yifangda[3])
print(u'博时悦楚纯债',bond_boshi[1],bond_boshi[2],bond_boshi[3])
print(u'招商招盛A',bond_zhaoshang[1],bond_zhaoshang[2],bond_zhaoshang[3])
print(u'南方多元',bond_nanfang[1],bond_nanfang[2],bond_nanfang[3])
print(u'中金金利A',bond_zhongjin[1],bond_zhongjin[2],bond_zhongjin[3])
print(u'天治可转债增强C',bond_tianzhi[1],bond_tianzhi[2],bond_tianzhi[3])

sns.plt.plot(bond_gongyin[0],label = u'Gongyin',color = 'b')
sns.plt.plot(bond_yifangda[0],label = 'Yifangda',color = 'c')
sns.plt.plot(bond_boshi[0],label = 'Boshi', color = 'g')
sns.plt.plot(bond_zhaoshang[0],label = 'Zhaoshang',color = 'k')
sns.plt.plot(bond_nanfang[0],label = 'Nanfang', color = 'm')
sns.plt.plot(bond_zhongjin[0],label = 'Zhongjin', ls=':', color = 'b')
sns.plt.plot(bond_tianzhi[0],label = 'Tianzhi',ls=':', color = 'c')
sns.plt.legend()
sns.plt.show()

money_gongyin = process_money_data(u'fund/money/工银瑞信货币.xlsx')
money_yifangda = process_money_data(u'fund/money/易方达易理财.xlsx')
money_boshi = process_money_data(u'fund/money/博时合鑫货币.xlsx')
money_zhaoshang = process_money_data(u'fund/money/招商招钱宝B.xlsx')
money_nanfang = process_money_data(u'fund/money/南方现金增利A.xlsx')
money_taixin = process_money_data(u'fund/money/泰信天天收益B.xlsx')
money_nuode = process_money_data(u'fund/money/诺德货币B.xlsx')
money_zhongjin = process_money_data(u'fund/money/中金现金管家B.xlsx')
money_tianzhi = process_money_data(u'fund/money/天治天得利货币.xlsx')
money_huarun = process_money_data(u'fund/money/华润元大现金收益B.xlsx')

sns.plt.plot(money_gongyin[0],label = u'Gongyin',color = 'b')
sns.plt.plot(money_yifangda[0],label = 'Yifangda',color = 'c')
sns.plt.plot(money_boshi[0],label = 'Boshi', color = 'g')
sns.plt.plot(money_zhaoshang[0],label = 'Zhaoshang',color = 'k')
sns.plt.plot(money_nanfang[0],label = 'Nanfang', color = 'm')
sns.plt.plot(money_taixin[0],label = 'Taixin', ls=':', color = 'b')
sns.plt.plot(money_nuode[0],label = 'Nuode', ls=':', color = 'c')
sns.plt.plot(money_zhongjin[0],label = 'Zhongjin', ls=':', color = 'g')
sns.plt.plot(money_tianzhi[0],label = 'Tianzhi',ls=':', color = 'k')
sns.plt.plot(money_huarun[0],label = 'Huarun', ls=':', color = 'm')
sns.plt.legend()
sns.plt.show()

print(u'工银瑞信货币',money_gongyin[2],money_gongyin[3])
print(u'易方达易理财',money_yifangda[2],money_yifangda[3])
print(u'博时合鑫货币',money_boshi[2],money_boshi[3])
print(u'招商招钱宝B',money_zhaoshang[2],money_zhaoshang[3])
print(u'南方现金增利A',money_nanfang[2],money_nanfang[3])
print(u'泰信天天收益B',money_taixin[2],money_taixin[3])
print(u'诺德货币B',money_nuode[2],money_nuode[3])
print(u'中金现金管家B',money_zhongjin[2],money_zhongjin[3])
print(u'天治天得利货币',money_tianzhi[2],money_tianzhi[3])
print(u'华润元大现金收益B',money_huarun[2],money_huarun[3])

