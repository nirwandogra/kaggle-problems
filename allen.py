import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/nirwandogra/Downloads/training_set.tsv', error_bad_lines=False, header = 0, delimiter = "\t" , index_col = [0])
cols = [ 'correctAnswer', 'question','answerA', 'answerB', 'answerC', 'answerD']
df = df[cols]
df['correctAnswer'] = df['correctAnswer'].map( lambda x: ord(x) )

non_numeric_features = ['question','answerA', 'answerB', 'answerC', 'answerD']
for non_numeric_feature in non_numeric_features:
    le = LabelEncoder()
    le.fit(list(df[non_numeric_feature]))
    df[non_numeric_feature] = le.transform(df[non_numeric_feature])

print df.head(1)
train_data = df.values
reg = LogisticRegression()
forest = RandomForestClassifier(n_estimators = 100)
reg = reg.fit(train_data[0::,1::] , train_data[0::,0])
forest = forest.fit(train_data[0::,1::] , train_data[0::,0])
rng = np.random.RandomState(0)
X = rng.rand(1, 5)
print len(train_data[0,1::])
output = reg.predict(X)

test_data = pd.read_csv('/Users/nirwandogra/Downloads/validation_set.tsv', error_bad_lines=False, header = 0, delimiter = "\t", index_col = [0])
cols = [ 'question','answerA', 'answerB', 'answerC', 'answerD']
test_data = test_data[cols]

non_numeric_features = ['question','answerA', 'answerB', 'answerC', 'answerD']
for non_numeric_feature in non_numeric_features:
    le = LabelEncoder()
    le.fit(list(test_data[non_numeric_feature]))
    test_data[non_numeric_feature] = le.transform(test_data[non_numeric_feature])

test_data_x = test_data.values
test_data['correctAnswer'] = forest.predict(test_data_x[0::,0::])
test_data['correctAnswer'] = test_data['correctAnswer'].map( lambda x: chr(x) )
print test_data['correctAnswer']
test_data[['correctAnswer']].to_csv('allenSolution.csv')
