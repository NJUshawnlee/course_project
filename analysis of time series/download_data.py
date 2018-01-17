import urllib.request
import pandas as pd
from pandas import DataFrame
import numpy as np


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

def get_returns(index,start,end):
    data=get_history_data(index,start,end)
    returns=np.log([(float(data[0][i][3]))/float(data[0][i][7]) for i in range((data[0]).__len__()-1,0,-1)])
    return returns

if __name__ == "__main__":
    returns=get_returns('sh000001', '20100101', '20170331')
    returns_csv=DataFrame(returns)
    returns_csv.to_csv('log_returns.csv')
