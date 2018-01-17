import numpy as np
import random
import math
import seaborn as sns


alpha = 1.3 #1.0859491161889947
beta = 0 #-0.056147643553253121
mu = 0 #0.29399373120070227
sigma = 1 #0.58594911618899481

a = math.pi/2
V = np.random.uniform(-a,a,10000)
W = np.array([random.expovariate(1) for i in range(0,10000)])
B = np.zeros(10000)
B [:]= (math.atan(beta*(math.tan(a*alpha))))/alpha
S = np.zeros(10000)
S[:] = (1+(beta**2)*(math.tan(a*alpha))**2)**(1/(2*alpha))


X = S*np.sin(alpha*(V+B))/((np.cos(V))**(1/alpha))*((np.cos(V-alpha*(V+B)))/W)**((1-alpha)/alpha)

Y = sigma*X+mu
print(Y)
sns.distplot(X, bins=100,label='stable')
sns.plt.legend()
sns.plt.show()







