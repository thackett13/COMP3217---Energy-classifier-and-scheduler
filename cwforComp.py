#*import csv
#for x in range(9999):
    

#with open('C:/Users/Tom/OneDrive/Desktop/cw/TrainingData.txt','r') as file:
 #       reader = csv.reader(file)
 #       for row in reader:
  #          print(row)


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
result = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/TrainingData.txt')

x = result.drop('24',axis = 1)
y = result['24']
print(x)
print(y)
x['sum'] = x.sum(axis=1)
print(x)
x['average'] = x['sum'].div(24)
print(x)
x['T'] = 0
x = x[['average','T']]
print(x)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.05)
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,y_train)
y_pred = svclassifier.predict(x_test)
print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", metrics.accuracy_score(y_test,y_pred)) 

#Read in actual test data
test = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/TestingData.txt')
print(test.shape)
print(test.head())
print(test)
test['sum'] = test.sum(axis = 1)
print(test)
test['average'] = test['sum'].div(24)
print(test)
test['T'] = 0
test = test[['average','T']]
print(test)
y_pred = svclassifier.predict(test)
print(y_pred)
