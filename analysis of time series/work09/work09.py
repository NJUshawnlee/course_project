__author__ = 'Shawn Li'

import tushare as ts
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


money_supply = ts.get_money_supply_bal()
gdp_year = ts.get_gdp_year()
gdp_for = ts.get_gdp_for()

length = len(money_supply)


def float_array(array):
    float_arr = np.array(array)
    float_arr = float_arr.astype(np.float64)
    return float_arr

m2 = float_array(money_supply['m2'])[0:length]
m1 = float_array(money_supply['m1'])[0:length]
m0 = float_array(money_supply['m0'])[0:length]
cd = float_array(money_supply['cd'])[0:length]

gdp = float_array(gdp_year['gdp'])[0:length]
pi = float_array(gdp_year['pi'])[0:length]
si = float_array(gdp_year['si'])[0:length]
ti = float_array(gdp_year['ti'])[0:length]

end_for = float_array(gdp_for['end_for'])[0:length-1]
asset_for = float_array(gdp_for['asset_for'])[0:length-1]
goods_for = float_array(gdp_for['goods_for'])[0:length-1]

end_for = np.insert(arr = end_for,obj = 0 ,values= 64.60)
asset_for = np.insert(arr = asset_for,obj = 0 ,values= 42.20)
goods_for = np.insert(arr = goods_for,obj = 0 ,values= -6.80)

\
sample1= (np.array([m2[1:length-1],m1[1:length-1],m0[1:length-1],cd[1:length-1],gdp[1:length-1]
                    ,gdp[2:],pi[1:length-1],si[1:length-1],ti[1:length-1],end_for[1:length-1],
                    asset_for[1:length-1],goods_for[1:length-1]])).T
target1 = gdp[0:length-2]
prediction_sample1 = (np.array([m2[0],m1[0],m0[0],cd[0],gdp[0],gdp[1],pi[0],si[0],
                        ti[0],end_for[0],asset_for[0],goods_for[0]])).T


sample2 = (np.array([m2[2:length-1],m1[2:length-1],m0[2:length-1],cd[2:length-1],gdp[2:length-1],
                     gdp[3:],pi[2:length-1],si[2:length-1],ti[2:length-1],end_for[2:length-1],
                     asset_for[2:length-1],goods_for[2:length-1]])).T
target2 = gdp[1:length-2]
prediction_sample2 = (np.array([m2[1],m1[1],m0[1],cd[1],gdp[1],gdp[2],pi[1],si[1],
                        ti[1],end_for[1],asset_for[1],goods_for[1]])).T

sample3 = (np.array([m2[3:length-1],m1[3:length-1],m0[3:length-1],cd[3:length-1],gdp[3:length-1],
                     gdp[4:],pi[3:length-1],si[3:length-1],ti[3:length-1],end_for[3:length-1],
                     asset_for[3:length-1],goods_for[3:length-1]])).T
target3 = gdp[2:length-2]
prediction_sample3 = (np.array([m2[2],m1[2],m0[2],cd[2],gdp[2],gdp[3],pi[2],si[2],
                        ti[2],end_for[2],asset_for[2],goods_for[2]])).T


def BP(sample,target,prediction_sample):
    sample_fit = MinMaxScaler().fit(sample)
    sample_transformed = sample_fit.transform(sample)
    target_fit = MinMaxScaler().fit(target)
    target_transformed = target_fit.transform(target)
    regressor = MLPRegressor(hidden_layer_sizes = 20, activation = 'relu', max_iter = 200, tol = 1e-4)
    fit_model = regressor.fit(sample_transformed,target_transformed)
    prediction_transformed =  sample_fit.transform(prediction_sample)
    prediction = target_fit.inverse_transform(fit_model.predict(prediction_transformed))
    return prediction

list1 = [0 for i in range(10000)]
for i in range(10000):
    list1[i] = BP(sample1,target1,prediction_sample1)
list1 = np.array(list1)
predict_2017 = np.mean(list1)

list2 = [0 for i in range(10000)]
for i in range(10000):
    list2[i] = BP(sample2,target2,prediction_sample2)
list2 = np.array(list2)
predict_2016 = np.mean(list2)

list3 = [0 for i in range(10000)]
for i in range(10000):
    list3[i] = BP(sample3,target3,prediction_sample3)
list3 = np.array(list3)
predict_2015 = np.mean(list3)


length = len(gdp)
gdp_list = [0 for i in range(length+1)]
for i in range(length-2):
    gdp_list[i] = gdp[length-i-1]

gdp_list[length-2] = predict_2015
gdp_list[length-1] = predict_2016
gdp_list[length] = predict_2017
gdp_predict = np.array(gdp_list)

gdp_reality = [0 for i in range(length)]
for i in range(length):
    gdp_reality[i] = gdp[length-i-1]

print(predict_2017)
print(predict_2016)
print(predict_2015)

sns.plt.plot(gdp_reality[-4:],marker='o',label='reality')
sns.plt.plot(gdp_predict[-5:],marker='o',ls=':', label='predict')
sns.plt.legend()
sns.plt.show()

