__author__ = 'Shawn Li'
#-*- coding: UTF-8 -*-

from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.sandbox.stats.runs import runstest_1samp
import download_data

def get_log_return(file,length):
    data = pd.read_csv(file,encoding='gbk')
    returns = np.array(data[u'收盘价']/data[u'前收盘价'])[-length:]
    log_returns = np.log(returns)
    return log_returns

SH = download_data.get_returns('sh000001', '20100101', '20170413')
SZ = download_data.get_returns('sz399001', '20100101', '20170413')
length1 = len(SZ)

HSI = get_log_return('HSI.CSV',length1)
DJONES = get_log_return('DJONES.CSV',length1)
SP500 = get_log_return('SP500.CSV',length1)
NASDAQ = get_log_return('NASDAQ.CSV', length1)
N225 = get_log_return('N225.CSV', length1)

def autocor_test(data):
    lag=int((len(data))**0.5)
    acf_result = acf(data,nlags=lag,qstat=True,alpha=0.05)
    runstest_result = runstest_1samp(data,cutoff='mean')
    plot_acf(data,lags=lag,alpha=0.05)
    sns.plt.ylim(-0.15,0.15)
    sns.plt.show()
    return (acf_result,runstest_result)

SH_result = autocor_test(SH)
SZ_result = autocor_test(SZ)
HSI_result = autocor_test(HSI)
DJONES_result = autocor_test(DJONES)
SP500_result = autocor_test(SP500)
NASDAQ_result = autocor_test(NASDAQ)
N225_result = autocor_test(N225)



np.savetxt('sh_q.csv',SH_result[0][2])
np.savetxt('sh_p.csv',SH_result[0][3])
np.savetxt('sz_q.csv',SZ_result[0][2])
np.savetxt('sz_p.csv',SZ_result[0][3])
np.savetxt('hsi_q.csv',HSI_result[0][2])
np.savetxt('hsi_p.csv',HSI_result[0][3])
np.savetxt('djones_q.csv',DJONES_result[0][2])
np.savetxt('djones_p.csv',DJONES_result[0][3])
np.savetxt('sp500_q.csv',SP500_result[0][2])
np.savetxt('sp500_p.csv',SP500_result[0][3])
np.savetxt('nasdaq_q.csv',NASDAQ_result[0][2])
np.savetxt('nasdaq_p.csv',NASDAQ_result[0][3])

print(SH_result)
print(SZ_result)
print(HSI_result)
print(DJONES_result)
print(SP500_result)
print(NASDAQ_result)
print(N225_result)





