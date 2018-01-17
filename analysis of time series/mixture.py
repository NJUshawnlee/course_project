import download_data
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def draw_mixture(index,start,end):
    returns=download_data.get_returns(index, start, end)
    length=len(returns)
    returns=returns.reshape((length, 1))
    aic=[0 for i in range(4)]
    bic=[0 for i in range(4)]
    for i in range(2,6):
        gmm=GaussianMixture(n_components=i,tol=1e-6,covariance_type='full')
        fgmm=gmm.fit(returns)
        aic[i-2]=fgmm.aic(returns)
        bic[i-2]=fgmm.bic(returns)
    k1=int(aic.index(min(aic)))+2
    k2=int(bic.index(min(bic)))+2
    print (k1)
    print (k2)
    if k1==k2:
        gmm=GaussianMixture(n_components=k1,tol=1e-6,covariance_type='full')
        fgmm=gmm.fit(returns)
        sample1=((fgmm.sample(10000000))[0].reshape(1, 10000000))
        sns.distplot(sample1, label='Gaussian',hist=None)
        sns.distplot(returns, bins=100, label='Empirical')
        sns.plt.legend()
        sns.plt.title('GaussianMixture')
        sns.plt.show()
    else:
        gmm1=GaussianMixture(n_components=k1,tol=1e-5,covariance_type='full')
        fgmm1=gmm1.fit(returns)
        gmm2=GaussianMixture(n_components=k2,tol=1e-5,covariance_type='full')
        fgmm2=gmm2.fit(returns)
        sample1=((fgmm1.sample(10000000))[0].reshape(1, 10000000))
        sample2=((fgmm2.sample(10000000))[0].reshape(1, 10000000))
        sns.distplot(sample1, label='Gaussian-AIC',hist=None)
        sns.distplot(sample2, label='Gaussian-BIC',hist=None)
        sns.distplot(returns, bins=100, label='Empirical')
        sns.plt.legend()
        sns.plt.title('GaussianMixture')
        sns.plt.show()


if __name__ == "__main__":
    draw_mixture('sh000001', '20070101', '20170331')
