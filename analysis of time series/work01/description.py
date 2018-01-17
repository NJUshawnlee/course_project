__author__ = 'Shawn Li'

import numpy as np
import stats as sts
import download_data
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


def data_description(index,start,end):
    returns=download_data.get_returns(index, start, end)
    print('个数：',len(returns))
    print('平均值:',np.mean(returns))
    print('中位数:',np.median(returns))
    print('上四分位数',sts.quantile(returns,p=0.25))
    print('下四分位数',sts.quantile(returns,p=0.75))
    #离散趋势的度量
    print('最大值:',np.max(returns))
    print('最小值:',np.min(returns))
    print('极差:',np.max(returns)-np.min(returns))
    print('四分位差',sts.quantile(returns,p=0.75)-sts.quantile(returns,p=0.25))
    print('标准差:',np.std(returns))
    print('方差:',np.var(returns))
    print('离散系数:',np.std(returns)/np.mean(returns))
    #偏度与峰度的度量
    print('偏度:',sts.skewness(returns))
    print('峰度:',sts.kurtosis(returns))
    print(st.kstest(returns,'norm'))
    length=len(returns)
    sns.distplot(returns, bins=100, label='Empirical')
    sns.plt.legend()
    sns.plt.title('Empirical')
    sns.plt.show()

if __name__ == "__main__":
    data_description('sh000001', '20100101', '20170331')
