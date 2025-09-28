#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""data collection and analysis 
PIMA diabetes dataset"""

#loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/Users/vaishnaviawadhiya/minor project/diabetes.csv')

#printing the first 5 rows of dataset
#print(diabetes_dataset.head())

#number of rows and colums in this dataset
#print(diabetes_dataset.shape)

#getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#seperating the data and labels 
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']
#print(x)
#print(y)

#Data Standardization
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
#print(standardized_data)

x = standardized_data
y = diabetes_dataset['Outcome']
#print(x)
#print(y)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

#training the model
classifier = svm.SVC(kernel='linear')

#training the support vector machine classifer
classifier.fit(x_train, y_train)

#model evaluration
#accuracy score

#accuracy score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

#making a prediction system
input_data = (1,85,66,29,0,26.6,0.351,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

#saving the trained model
import pickle
filename = 'trained model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load (open ('trained model.sav', 'rb'))

#input_data = (1,85,66,29,0,26.6,0.351,31)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray (input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape (1, -1)
prediction = loaded_model.predict (input_data_reshaped)


print(prediction)  

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
  