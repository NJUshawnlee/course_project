import download_data
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

def draw_lognorm(index,start,end):
    returns=download_data.get_returns(index, start, end)
    fit_data=st.norm(returns.mean(), returns.std()).rvs(10000000)
    sns.distplot(fit_data, label='log_norm',hist=None)
    length=len(returns)
    sns.distplot(returns, bins=100, label='Empirical')
    sns.plt.legend()
    sns.plt.title('LogNorm')
    sns.plt.show()

if __name__ == "__main__":
    draw_lognorm('sh000001', '20150101', '20170329')
