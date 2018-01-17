import pymysql.cursors

# 连接数据库
connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='liguohushen',
    db='world',
    charset='utf8'
)
