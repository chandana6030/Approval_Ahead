import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


loan_dataset = pd.read_csv('loan status prediction dataset.csv')


type(loan_dataset)


# printing the first 5 rows of the dataframe
loan_dataset.head()

# number of rows and columns
loan_dataset.shape

# number of missing values in each column
loan_dataset.isnull().sum()

# dropping the missing values
loan_dataset = loan_dataset.dropna()

# number of missing values in each column
loan_dataset.isnull().sum()

# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# dependent values
loan_dataset['Dependents'].value_counts()

# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

loan_dataset.dtypes

# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

loan_dataset.head()

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)

print(X.shape, X_train.shape, X_test.shape)

classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)

from sklearn.metrics import accuracy_score
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


print('Accuracy on training data : ', (training_data_accuracy)*100)
# accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data : ', (test_data_accuray)*100)

with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

