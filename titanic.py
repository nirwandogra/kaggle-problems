import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
df = pd.read_csv('train.csv', index_col = [0])
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
 
df['AgeFill'] = df['Age']
df.head()
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

df['Fares'] = df['Fare']
avg_fares = df['Fare'].median()
df.loc[ (df['Fare'].isnull()),\
                'Fares'] = avg_fares

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df = df.drop(['Fare'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.dropna()
train_data = df.values

reg = LogisticRegression()
rng = np.random.RandomState(0)
reg = reg.fit(train_data[0::,1::],train_data[0::,0])
print reg

test_data  = pd.read_csv('test.csv', index_col = [0])
test_data['Gender'] = test_data['Sex'].map( lambda x: x[0].upper() )
test_data['Gender'] = test_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_data['AgeFill'] = test_data['Age']
for i in range(0, 2):
    for j in range(0, 3):
        test_data.loc[ (test_data.Age.isnull()) & (test_data.Gender == i) & (test_data.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

test_data['Fares'] = test_data['Fare']

avg_fares = test_data['Fare'].median()
test_data.loc[ (test_data['Fare'].isnull()),\
                'Fares'] = avg_fares

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']
test_data['Age*Class'] = test_data.AgeFill * test_data.Pclass
test_data = test_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
test_data = test_data.drop(['Fare'], axis=1)
test_data = test_data.drop(['Age'], axis=1)

test_data_x = test_data.values[0::,0::]
xx = reg.predict(test_data_x)
test_data['Survived']=xx
print  test_data[['Survived']]
test_data['Survived'] = test_data['Survived'].astype(np.int)
test_data[['Survived']].to_csv('myoutput.csv')
