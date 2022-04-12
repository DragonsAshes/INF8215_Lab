import pandas as pd

df = pd.read_csv('./data/beans_train.csv')

keys = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']
dic = {key: None for key in keys}

for key in dic:
    dic[key] = df.loc[df['class'] == key, df.columns != 'class']

# print stats for each column class by class
for key in dic:
    print('\n','='*10, key, '='*10)
    for column in dic[key]:
        print(column, dic[key][column].max(), dic[key][column].min(), dic[key][column].mean())

# print stats for class column by column
for column in dic['SIRA']:
    print('\n','='*10,column, '='*10)
    for key in dic:
        print(key, dic[key][column].max(), dic[key][column].min(), dic[key][column].mean())
