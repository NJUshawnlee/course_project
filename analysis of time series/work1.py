import urllib.request
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from scipy.stats import levy_stable
import levy
import pandas as pd
from pandas import DataFrame

def get_page(url): #获取页面数据

    req= urllib.request.Request(url,headers={
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
        })
    opener= urllib.request.urlopen(req)
    page= opener.read()
    return page

def get_history_data(index,start,end):

    """
    :param index: for example,'sh000001' 上证指数
    :return :
    """
    index_type = index[0:2]
    index_id  =index[2:]
    if index_type=='sh':
        index_id='0'+index_id
    if index_type=='sz':
        index_id='1'+index_id
    url ='http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER'%(index_id,start,end)

    page = get_page(url).decode('gb2312')
    page = page.split('\r\n')
    col_info = page[0].split(',')
    index_data = page[1:]
    index_data = [x.replace("'",'') for x in index_data]
    index_data = [x.split(',') for x in index_data]


    index_data=index_data[0:index_data.__len__()-1]   #最后一行为空，需要去掉
    pos1 = col_info.index('涨跌幅')
    pos2 = col_info.index('涨跌额')
    posclose=col_info.index('收盘价')
    index_data[index_data.__len__()-1][pos1]=0     
    index_data[index_data.__len__()-1][pos2]=0
    for i in range(0,index_data.__len__()-1):       
        if index_data[i][pos2]=='None':
            index_data[i][pos2]=float(index_data[i][posclose])-float(index_data[i+1][posclose])
        if index_data[i][pos1]=='None':
            index_data[i][pos1]=(float(index_data[i][posclose])-float(index_data[i+1][posclose]))/float(index_data[i+1][posclose])
    return [index_data,col_info]

data=get_history_data('sh000001', '20160101', '20170329')
returns=np.log([(float(data[0][i][3]))/float(data[0][i][7]) for i in range((data[0]).__len__()-1,0,-1)])
returns_csv=DataFrame(returns)
returns_csv.to_csv('log_returns.csv')
length=len(returns)
fit_data=st.norm(returns.mean(), returns.std()).rvs(length)
result=levy.fit_levy(returns)
print (result)
'''fit_data2=levy.random(alpha=stable_para[0],beta=stable_para[1],mu=stable_para[2],sigma=stable_para[3],shape=(1,length),par=1)'''

returns=returns.reshape((length, 1))
aic=[0 for i in range(4)]
bic=[0 for i in range(4)]
for i in range(2,6):
    gmm=GaussianMixture(n_components=i,tol=1e-12,covariance_type='full')
    fgmm=gmm.fit(returns)
    aic[i-2]=fgmm.aic(returns)
    bic[i-2]=fgmm.bic(returns)
k1=int(aic.index(min(aic)))+2
k2=int(bic.index(min(bic)))+2
print (k1)
print (k2)
'''result = levy.fit_levy(levy.random(1.5, 0.5, 0.0, 1.0, 100000))
print (result)
print(levy.random(1.5, 0.5, 0.0, 1.0, 1000000).mean())'''
gmm1=GaussianMixture(n_components=k1,tol=1e-12,covariance_type='full')
gmm2=GaussianMixture(n_components=k2,tol=1e-12,covariance_type='full')
sample1=((gmm1.fit(returns).sample(100000000))[0].reshape(1, 100000000))
sample2=((gmm2.fit(returns).sample(100000000))[0].reshape(1, 100000000))
sns.distplot(sample1, bins=500, label='Gaussian1',hist=None)
sns.distplot(sample2, bins=500, label='Gaussian2',hist=None)
sns.distplot(returns, bins=length, label='Reality',hist=None)
sns.plt.legend()
sns.plt.show()

'''dgmm = gmm.fit(returns).sample(1000)
samples = dgmm[0].reshape((1,1000))
sns.distplot(samples,bins=1000)'''
'''sns.distplot(returns,bins=length)'''
'''sns.distplot(fit_data, bins=length,label='Norm')'''
'''sns.distplot(levy.random(1.5, 0.5, 0.0, 1.0, 1000000) , bins=100, label='Stable')'''
'''alpha, beta, mu, sigma = result[0], result[1], result[2], result[3]
x = np.linspace(-2.0,2.0, 40)'''
'''sns.plt.plot(x/80, levy.levy(x, alpha, beta, mu, sigma)*len(returns)/40, 'k-', color='blue', lw=2, label='levy_stable pdf')'''
'''plt.hist(returns, bins =100, color = 'r',alpha=0.5,rwidth= 0.9, normed=True)'''
'''sns.plt.xlim([-0.15, 0.15])'''
'''sns.plt.legend()
sns.plt.show()'''
'''ax.plot(x, levy_stable.pdf(x, alpha, beta),
      'r-', lw=5, alpha=0.6, label='levy_stable pdf')'''

'''alpha = 1.3 #1.0859491161889947
beta = 0 #-0.056147643553253121
mu = 0 #0.29399373120070227
sigma = 1 #0.58594911618899481

a = math.pi/2
V = np.random.uniform(-a,a,10000)
W = np.random.exponential(1,10000)
B = np.zeros(10000)
B [:]= (math.atan(beta*(math.tan(a*alpha))))/alpha
S = np.zeros(10000)
S[:] = (1+(beta**2)*(math.tan(a*alpha))**2)**(1/(2*alpha))


X = S*np.sin(alpha*(V+B))/((np.cos(V))**(1/alpha))*((np.cos(V-alpha*(V+B)))/W)**((1-alpha)/alpha)

Y = sigma*X+mu
print(Y.mean())
sns.distplot(Y, bins=10000,label='stable')
sns.distplot(fit_data, bins=length,label='Norm')
sns.plt.xlim([-0.15, 0.15])
sns.plt.legend()
sns.plt.show()'''
