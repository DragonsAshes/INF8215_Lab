import pandas as pd

df = pd.read_csv('./data/beans_train.csv')

keys = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']
dic = {key: None for key in keys}

print(dic)

for key in dic:
    dic[key] = df.loc[df['class'] == key, df.columns != 'class']

for key in dic:
    print('\n','='*10, key, '='*10)
    for column in dic[key]:
        print(column, dic[key][column].max(), dic[key][column].min(), dic[key][column].mean())

