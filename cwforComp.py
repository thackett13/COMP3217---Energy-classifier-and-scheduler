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
from lpsolve55 import *

numpy.set_printoptions(threshold=sys.maxsize)
result = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/TrainingData.txt')
tasks = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/Tasks2.txt')
print(tasks)
x = result.drop('24',axis = 1)
y = result['24']
print(x)
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,y_train)
y_pred = svclassifier.predict(x_test)
print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", metrics.accuracy_score(y_test,y_pred)) 

#Read in actual test data
test = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/TestingData.txt')
test1 = test
#print(test.shape)
#print(test.head())
print(test)



y_pred = svclassifier.predict(test)
print(y_pred)
test1['24'] = 0
t = 0
for x in range(100):
    if(y_pred[x] == 1):
        t = t + 1
print(t)
Tasks = pandas.read_csv('C:/Users/Tom/OneDrive/Desktop/cw/COMP3217CW2Tasks.txt')

Task1 = []
Task2 = []
Task3 = []
Task4 = []
Task5 = []
#Arrays of energy demand 
A = []
B = []
C = []
D = []
E = []
#Arrays of start time
A1 = []
B1 = []
C1 = []
D1 = []
E1 = []
#Arrays of deadline
A2 = []
B2 = []
C2 = []
D2 = []
E2 = []

#Times:



for x in range(10):
    Task1.append(Tasks.loc[Tasks['User & Task ID']== 'user1_task'+str(x+1)])
    Task2.append(Tasks.loc[Tasks['User & Task ID'] =='user2_task'+str(x+1)])
    Task3.append(Tasks.loc[Tasks['User & Task ID'] =='user3_task'+str(x+1)])
    Task4.append(Tasks.loc[Tasks['User & Task ID'] =='user4_task'+str(x+1)])
    Task5.append(Tasks.loc[Tasks['User & Task ID'] =='user5_task'+str(x+1)])
for x in range(10):
    A.append(Task1[x]['Energy Demand'])
    B.append(Task2[x]['Energy Demand'])
    C.append(Task3[x]['Energy Demand'])
    D.append(Task4[x]['Energy Demand'])
    E.append(Task5[x]['Energy Demand'])

    A1.append(Task1[x]['Ready Time'])
    B1.append(Task2[x]['Ready Time'])
    C1.append(Task3[x]['Ready Time'])
    D1.append(Task4[x]['Ready Time'])
    E1.append(Task5[x]['Ready Time'])

    A2.append(Task1[x]['Deadline'])
    B2.append(Task2[x]['Deadline'])
    C2.append(Task3[x]['Deadline'])
    D2.append(Task4[x]['Deadline'])
    E2.append(Task5[x]['Deadline'])



for i in range(100):

    if(y_pred[i] == 1):
            T0 = test1.iloc[i,0]
            T1 = test1.iloc[i,1]
            T2 = test1.iloc[i,2]
            T3 = test1.iloc[i,3]
            T4 = test1.iloc[i,4]
            T5 = test1.iloc[i,5]
            T6 = test1.iloc[i,6]
            T7 = test1.iloc[i,7]
            T8 = test1.iloc[i,8]
            T9 = test1.iloc[i,9]
            T10 = test1.iloc[i,10]
            T11 = test1.iloc[i,11]
            T12 = test1.iloc[i,12]
            T13 = test1.iloc[i,13]
            T14 = test1.iloc[i,14]
            T15 = test1.iloc[i,15]
            T16 = test1.iloc[i,16]
            T17 = test1.iloc[i,17]
            T18 = test1.iloc[i,18]
            T19 = test1.iloc[i,19]
            T20 = test1.iloc[i,20]
            T21 = test1.iloc[i,21]
            T22 = test1.iloc[i,22]
            T23 = test1.iloc[i,23]
            test1.iloc[i,24] = 1
            fileName = 'test'+str(i)+'.LP'
            f = open(fileName,'w')
            #z = open('C:/Users/Tom/OneDrive/Desktop/cw/Tasks2.txt','w')


            print(T0)

            f.write('min: a b c d e;')

            f.write('\n')

            print(' ' +str(T1))

            #f.write('c ='+str(T0)+' C9_0 +'+str(T1)+' C9_1+'+str(T1)+' D0_1+'+str(T2)+' C3_2+'+str(T2)+' C9_2+'+str(T2)+' D0_2+'+str(T2)+' E3_2+'+str(T2)+' E6_2+'+str(T3)+' B6_3+'+str(T3)+' C3_3+'+str(T3)+' C9_3+'+str(T3)+' D0_3+'+str(T3)+' E3_3+'+str(T3)+' E6_3+'+str(T4)+' A6_4+'+str(T4)+' B6_4+'+str(T4)+' C3_4+'+str(T4)+' C9_4+'+str(T4)+' D0_4+'+str(T4)+' D9_4+'+str(T4)+' E0_4+'+str(T4)+' E2_4+'+str(T4)+' E3_4+'+str(T4)+' E6_4+'+str(T5)+' A6_5+'+str(T5)+' B1_5+'+str(T5)+' B2_5+'+str(T5)+' B6_5+'+str(T5)+' C3_5+'+str(T5)+' C9_5+'+str(T5)+' D0_5+'+str(T5)+' D9_5+'+str(T5)+' E0_5+'+str(T5)+' E2_5+'+str(T5)+' E3_5+'+str(T5)+' E6_5+'+str(T6)+' A4_6+'+str(T6)+' A6_6+'+str(T6)+' B1_6+'+str(T6)+' B2_6+'+str(T6)+' B3_6+'+str(T6)+' B6_6+'+str(T6)+' B9_6+'+str(T6)+' C3_6+'+str(T6)+' C9_6+'+str(T6)+' D0_6+'+str(T6)+' D9_6+'+str(T6)+' E0_6+'+str(T6)+' E2_6+'+str(T6)+' E3_6+'+str(T6)+' E5_6+'+str(T6)+' E6_6+'+str(T7)+'A4_7+'+str(T7)+' A6_7+'+str(T7)+' A8_7+'+str(T7)+' B1_7+'+str(T7)+' B2_7+'+str(T7)+' B3_7+'+str(T7)+' B6_7+'+str(T7)+' B9_7+'+str(T7)+' C3_7+'+str(T7)+' C8_7+'+str(T7)+' C9_7+'+str(T7)+' D0_7+'+str(T7)+' D9_7+'+str(T7)+' E0_7+'+str(T7)+' E2_7+'+str(T7)+' E3_7+'+str(T7)+' E5_7+'+str(T8)+' A4_8+'+str(T8)+' A6_8+'+str(T8)+' A8_8+'+str(T8)+' A9_8+'+str(T8)+' B1_8+'+str(T8)+' B2_8+'+str(T8)+' B3_8+'+str(T8)+' B6_8+'+str(T8)+' B9_8+'+str(T8)+' C3_8+'+str(T8)+' C8_8+'+str(T8)+' D0_8+'+str(T8)+' D8_8+'+str(T8)+' D9_8+'+str(T8)+' E0_8+'+str(T8)+' E2_8+'+str(T8)+' E3_8+'+str(T8)+' E5_8+'+str(T9)+' A4_9+'+str(T9)+' A6_9+'+str(T9)+' A8_9+'+str(T9)+' A9_9+'+str(T9)+' B1_9+'+str(T9)+' B2_9+'+str(T9)+' B3_9+'+str(T9)+' B6_9+'+str(T9)+'B9_9+'+str(T9)+'C3_9+'+str(T9)+' C8_9+'+str(T9)+' D8_9+'+str(T9)+' D9_9+'+str(T9)+' E0_9+'+str(T9)+' E2_9+'+str(T9)+' E3_9+'+str(T9)+' E5_9+'+str(T10)+' A4_10+'+str(T10)+' A6_10+'+str(T10)+' A8_10+'+str(T10)+' A9_10+'+str(T10)+' B1_10+'+str(T10)+' B2_10+'+str(T10)+' B3_10+'+str(T10)+' B6_10+'+str(T10)+' B9_10+'+str(T10)+' C3_10+'+str(T10)+' C5_10+'+str(T10)+' C8_10+'+str(T10)+' D8_10+'+str(T10)+' D9_10+'+str(T10)+' E0_10+'+str(T10)+' E2_10+'+str(T10)+' E3_10+'+str(T10)+' E5_10+'+str(T11)+' A4_11+'+str(T11)+' A8_11+'+str(T11)+' A9_11+'+str(T11)+' B0_11+'+str(T11)+' B1_11+'+str(T11)+' B2_11+'+str(T11)+' B3_11+'+str(T11)+' B6_11+'+str(T11)+' B9_11+'+str(T11)+' C2_11+'+str(T11)+' C3_11+'+str(T11)+' C5_11+'+str(T11)+' C8_11+'+str(T11)+' D1_11+'+str(T11)+' D3_11+'+str(T11)+' D8_11+'+str(T11)+' D9_11+'+str(T11)+' E0_11+'+str(T11)+' E2_11+'+str(T11)+' E3_11+'+str(T11)+' E5_11+'+str(T12)+' A3_12+'+str(T12)+' A4_12+'+str(T12)+' A7_12+'+str(T12)+' A8_12+'+str(T12)+' A9_12+'+str(T12)+' B0_12+'+str(T12)+' B2_12+'+str(T12)+' B3_12+'+str(T12)+' B6_12+'+str(T12)+' C2_12+'+str(T12)+' C3_12+'+str(T12)+' C5_12+'+str(T12)+' C8_12+'+str(T12)+' D1_12+'+str(T12)+'D2_12+'+str(T12)+' D3_12+'+str(T12)+' D7_12+'+str(T12)+' D8_12+'+str(T12)+' D9_12+'+str(T12)+' E0_12+'+str(T12)+' E2_12+'+str(T12)+'E3_12+'+str(T12)+'E5_12+'+str(T13)+' A3_13+'+str(T13)+' A7_13+'+str(T13)+' A8_13+'+str(T13)+' A9_13+'+str(T13)+' B0_13+'+str(T13)+' B2_13+'+str(T13)+' B3_13+'+str(T13)+' B6_13+'+str(T13)+' B8_13+'+str(T13)+' C2_13+'+str(T13)+' C3_13+'+str(T13)+' C4_13+'+str(T13)+' C5_13+'+str(T13)+' C8_13+'+str(T13)+' D1_13+'+str(T13)+' D2_13+'+str(T13)+' D3_13+'+str(T13)+' D7_13+'+str(T13)+' D8_13+'+str(T13)+' E0_13+'+str(T13)+' E2_13+'+str(T13)+' E3_13+'+str(T13)+' E5_13+'+str(T13)+' E7_13+'+str(T14)+' A3_14+'+str(T14)+' A7_14+'+str(T14)+' A8_14+'+str(T14)+' A9_14+'+str(T14)+' B0_14+'+str(T14)+' B2_14+'+str(T14)+' B3_14+'+str(T14)+' B6_14+'+str(T14)+' B8_14+'+str(T14)+' C2_14+'+str(T14)+' C3_14+'+str(T14)+' C4_14+'+str(T14)+' C5_14+'+str(T14)+' C8_14+'+str(T14)+' D1_14+'+str(T14)+' D2_14+'+str(T14)+' D3_14+'+str(T14)+' D7_14+'+str(T14)+' D8_14+'+str(T14)+' E0_14+'+str(T14)+' E2_14+'+str(T14)+' E3_14+'+str(T14)+' E5_14+'+str(T14)+' E7_14+'+str(T15)+' A3_15+'+str(T15)+' A7_15+'+str(T15)+' B0_15+'+str(T15)+' B2_15+'+str(T15)+'B3_15+'+str(T15)+' B6_15+'+str(T15)+' B8_15+'+str(T15)+' C1_15+'+str(T15)+' C2_15+'+str(T15)+' C3_15+'+str(T15)+' C4_15+'+str(T15)+' C5_15+'+str(T15)+' C8_15+'+str(T15)+' D1_15+'+str(T15)+' D2_15+'+str(T15)+' D3_15+'+str(T15)+' D7_15+'+str(T15)+' D8_15+'+str(T15)+' E0_15+'+str(T15)+'E2_15+'+str(T15)+'E3_15+'+str(T15)+'E5_15+'+str(T15)+' E7_15+'+str(T15)+' E8_15+'+str(T16)+' A3_16+'+str(T16)+' A7_16+'+str(T16)+' B0_16+'+str(T16)+' B2_16+'+str(T16)+' B3_16+'+str(T16)+' B6_16+'+str(T16)+' B8_16+'+str(T16)+' C1_16+'+str(T16)+' C3_16+'+str(T16)+' C4_16+'+str(T16)+' C5_16+'+str(T16)+' C8_16+'+str(T16)+' D1_16+'+str(T16)+' D2_16+'+str(T16)+' D3_16+'+str(T16)+' D4_16+'+str(T16)+' D7_16+'+str(T16)+' D8_16+'+str(T16)+' E0_16+'+str(T16)+' E2_16+'+str(T16)+' E3_16+'+str(T16)+' E4_16+'+str(T16)+' E5_16+'+str(T16)+' E7_16+'+str(T16)+' E8_16+'+str(T17)+' A3_17+'+str(T17)+' A7_17+'+str(T17)+' B0_17+'+str(T17)+' B2_17+'+str(T17)+' B3_17+'+str(T17)+' B6_17+'+str(T17)+' B8_17+'+str(T17)+' C1_17+'+str(T17)+' C3_17+'+str(T17)+'C5_17+'+str(T17)+'C8_17+'+str(T17)+' D1_17+'+str(T17)+' D2_17+'+str(T17)+' D4_17+'+str(T17)+' D7_17+'+str(T17)+' D8_17+'+str(T17)+' E0_17+'+str(T17)+' E4_17+'+str(T17)+' E5_17+'+str(T17)+' E7_17+'+str(T17)+' E8_17+'+str(T17)+' E9_17+'+str(T18)+' A1_18+'+str(T18)+' A3_18+'+str(T18)+' A5_18+'+str(T18)+' A7_18+'+str(T18)+' B0_18+'+str(T18)+' B2_18+'+str(T18)+' B3_18+'+str(T18)+' B5_18+'+str(T18)+' B6_18+'+str(T18)+' C1_18+'+str(T18)+' C5_18+'+str(T18)+' C8_18+'+str(T18)+' D1_18+'+str(T18)+' D2_18+'+str(T18)+' D4_18+'+str(T18)+' D7_18+'+str(T18)+' D8_18+'+str(T18)+' E0_18+'+str(T18)+' E1_18+'+str(T18)+' E4_18+'+str(T18)+' E5_18+'+str(T18)+' E8_18+'+str(T18)+' E9_18+'+str(T19)+' A1_19+'+str(T19)+' A2_19+'+str(T19)+' A3_19+'+str(T19)+' A5_19+'+str(T19)+' B0_19+'+str(T19)+' B2_19+'+str(T19)+' B3_19+'+str(T19)+' B4_19+'+str(T19)+' B5_19+'+str(T19)+' B6_19+'+str(T19)+' C1_19+'+str(T19)+' C8_19+'+str(T19)+' D1_19+'+str(T19)+' D2_19+'+str(T19)+' D5_19+'+str(T19)+' D7_19+'+str(T19)+' D8_19+'+str(T19)+' E0_19+'+str(T19)+' E1_19+'+str(T19)+' E4_19+'+str(T19)+' E8_19+'+str(T19)+' E9_19+'+str(T20)+' A0_20+'+str(T20)+' A1_20+'+str(T20)+' A2_20+'+str(T20)+' A3_20+'+str(T20)+' A5_20+'+str(T20)+' B0_20+'+str(T20)+' B2_20+'+str(T20)+' B3_20+'+str(T20)+' B5_20+'+str(T20)+' B6_20+'+str(T20)+' C0_20+'+str(T20)+' C1_20+'+str(T20)+' C7_20+'+str(T20)+'C8_20+'+str(T20)+'D1_20+'+str(T20)+'D5_20+'+str(T20)+'D8_20+'+str(T20)+'E0_20+'+str(T20)+'E1_20+'+str(T20)+'E4_20+'+str(T20)+'E8_20+'+str(T20)+' E9_20+'+str(T21)+' A0_21+'+str(T21)+' A1_21+'+str(T21)+' A2_21+'+str(T21)+' B0_21+'+str(T21)+' B2_21+'+str(T21)+' B5_21+'+str(T21)+' B6_21+'+str(T21)+' B7_21+'+str(T21)+' C0_21+'+str(T21)+' C1_21+'+str(T21)+' C6_21+'+str(T21)+'C7_21+'+str(T21)+' C8_21+'+str(T21)+' D5_21+'+str(T21)+' E1_21+'+str(T21)+' E4_21+'+str(T21)+' E8_21+'+str(T21)+' E9_21+'+str(T22)+' A0_22+'+str(T22)+' A1_22+'+str(T22)+' B0_22+'+str(T22)+' B2_22+'+str(T22)+' B6_22+'+str(T22)+' B7_22+'+str(T22)+' C0_22+'+str(T22)+' C6_22+'+str(T22)+' C7_22+'+str(T22)+' D5_22+'+str(T22)+' D6_22+'+str(T22)+' E1_22+'+str(T22)+'E4_22+'+str(T22)+' E8_22+'+str(T22)+' E9_22+'+str(T23)+' A0_23+'+str(T23)+' A1_23+'+str(T23)+' B2_23+'+str(T23)+' B6_23+'+str(T23)+' B7_23+'+str(T23)+' C0_23+'+str(T23)+' C6_23+'+str(T23)+' C7_23+'+str(T23)+' D5_23+'+str(T23)+' D6_23+'+ str(T23)+' E4_23+'+str(T23)+' E8_23+'+str(T23)+' E9_23')
            f.write('c ='+str(T0)+' C9_0+'+str(T1)+' C9_1+'+str(T2)+' C3_2+'+ str(T2)+' C9_2+'+ str(T3)+' C3_3+'+str(T4)+' C3_4+'+str(T4)+' C9_4+'+str(T5)+' C3_5+'+str(T5)+' C9_5+'+str(T6)+' C3_6+'+str(T6)+' C9_6+'+str(T7)+' C3_7+'+str(T7)+' C8_7+'+str(T7)+' C9_7+'+str(T8)+' C3_8+'+str(T8)+' C8_8+'+str(T9)+'C3_9+'+str(T9)+' C8_9+'+str(T10)+' C3_10+'+str(T10)+' C5_10+'+str(T10)+' C8_10+'+str(T11)+' C2_11+'+str(T11)+' C3_11+'+str(T11)+' C5_11+'+str(T11)+' C8_11+'+str(T12)+' C2_12+'+str(T12)+' C3_12+'+str(T12)+' C5_12+'+str(T12)+' C8_12+'+str(T13)+' C2_13+'+str(T13)+' C3_13+'+str(T13)+' C4_13+'+str(T13)+' C5_13+'+str(T13)+' C8_13+'+str(T14)+' C2_14+'+str(T14)+' C3_14+'+str(T14)+' C4_14+'+str(T14)+' C5_14+'+str(T14)+' C8_14+'+str(T15)+' C1_15+'+str(T15)+' C2_15+'+str(T15)+' C3_15+'+str(T15)+' C4_15+'+str(T15)+' C5_15+'+str(T15)+' C8_15+'+str(T16)+' C1_16+'+str(T16)+' C3_16+'+str(T16)+' C4_16+'+str(T16)+' C5_16+'+str(T16)+' C8_16+'+str(T17)+' C1_17+'+str(T17)+' C3_17+'+str(T17)+'C5_17+'+str(T17)+'C8_17+'+str(T18)+' C1_18+'+str(T18)+' C5_18+'+str(T18)+' C8_18+'+str(T19)+' C1_19+'+str(T19)+' C8_19+'+str(T20)+' C0_20+'+str(T20)+' C1_20+'+str(T20)+' C7_20+'+str(T20)+'C8_20+'+str(T21)+' C0_21+'+str(T21)+' C1_21+'+str(T21)+' C6_21+'+str(T21)+'C7_21+'+str(T21)+' C8_21+'+str(T22)+' C0_22+'+str(T22)+' C6_22+'+str(T22)+' C7_22+'+str(T23)+' C0_23+'+str(T23)+' C6_23+'+str(T23)+' C7_23')
            f.write(';')
            f.write('\n')
            f.write('d =' +str(T1)+' D0_1+'+str(T2)+' D0_2+'+str(T3)+' D0_3+'+str(T4)+' D0_4+'+str(T4)+' D9_4+'+str(T5)+' D0_5+'+str(T5)+' D9_5+'+str(T6)+' D0_6+'+str(T6)+' D9_6+'+str(T7)+' D0_7+'+str(T7)+' D9_7+'+str(T8)+' D0_8+'+str(T8)+' D8_8+'+str(T8)+' D9_8+'+str(T9)+' D8_9+'+str(T9)+' D9_9+'+str(T10)+' D8_10+'+str(T10)+' D9_10+'+str(T11)+' D1_11+'+str(T11)+' D3_11+'+str(T11)+' D8_11+'+str(T11)+' D9_11+'+str(T12)+' D1_12+'+str(T12)+'D2_12+'+str(T12)+' D3_12+'+str(T12)+' D7_12+'+str(T12)+' D8_12+'+str(T12)+' D9_12+'+str(T13)+' D1_13+'+str(T13)+' D2_13+'+str(T13)+' D3_13+'+str(T13)+' D7_13+'+str(T13)+' D8_13+'+str(T14)+' D1_14+'+str(T14)+' D2_14+'+str(T14)+' D3_14+'+str(T14)+' D7_14+'+str(T14)+' D8_14+'+str(T15)+' D1_15+'+str(T15)+' D2_15+'+str(T15)+' D3_15+'+str(T15)+' D7_15+'+str(T15)+' D8_15+'+str(T16)+' D1_16+'+str(T16)+' D2_16+'+str(T16)+' D3_16+'+str(T16)+' D4_16+'+str(T16)+' D7_16+'+str(T16)+' D8_16+'+str(T17)+' D1_17+'+str(T17)+' D2_17+'+str(T17)+' D4_17+'+str(T17)+' D7_17+'+str(T17)+' D8_17+'+str(T18)+' D1_18+'+str(T18)+' D2_18+'+str(T18)+' D4_18+'+str(T18)+' D7_18+'+str(T18)+' D8_18+'+str(T19)+' D1_19+'+str(T19)+' D2_19+'+str(T19)+' D5_19+'+str(T19)+' D7_19+'+str(T19)+' D8_19+'+str(T20)+'D1_20+'+str(T20)+'D5_20+'+str(T20)+'D8_20+'+str(T21)+' D5_21+'+str(T22)+' D5_22+'+str(T22)+' D6_22+'+str(T23)+' D5_23+'+str(T23)+' D6_23')
            f.write(';')
            f.write('\n')
            f.write('e ='+str(T2)+' E3_2+'+str(T2)+' E6_2+'+str(T3)+' E3_3+'+str(T3)+' E6_3+'+str(T4)+' E0_4+'+str(T4)+' E2_4+'+str(T4)+' E3_4+'+str(T4)+' E6_4+'+str(T5)+' E0_5+'+str(T5)+' E2_5+'+str(T5)+' E3_5+'+str(T5)+' E6_5+'+str(T6)+' E0_6+'+str(T6)+' E2_6+'+str(T6)+' E2_6+'+str(T6)+' E3_6+'+str(T6)+' E5_6+'+str(T6)+' E6_6+'+str(T7)+' E0_7+'+str(T7)+' E2_7+'+str(T7)+' E3_7+'+str(T7)+' E5_7+'+str(T8)+' E0_8+'+str(T8)+' E2_8+'+str(T8)+' E3_8+'+str(T8)+' E5_8+'+str(T9)+' E0_9+'+str(T9)+' E2_9+'+str(T9)+' E3_9+'+str(T9)+' E5_9+'+str(T10)+' E0_10+'+str(T10)+' E2_10+'+str(T10)+' E3_10+'+str(T10)+' E5_10+'+str(T11)+' E0_11+'+str(T11)+' E2_11+'+str(T11)+' E3_11+'+str(T11)+' E5_11+'+str(T12)+' E0_12+'+str(T12)+' E2_12+'+str(T12)+' E3_12+'+str(T12)+' E5_12+'+str(T13)+' E0_13+'+str(T13)+' E2_13+'+str(T13)+' E3_13+'+str(T13)+' E5_13+'+str(T13)+' E7_13+'+str(T14)+' E0_14+'+str(T14)+' E2_14+'+str(T14)+' E3_14+'+str(T14)+' E5_14+'+str(T14)+' E7_14+'+str(T15)+' E0_15+'+str(T15)+' E2_15+'+str(T15)+' E3_15+'+str(T15)+' E5_15+'+str(T15)+' E7_15+'+str(T15)+' E8_15+'+str(T16)+' E0_16+'+str(T16)+' E2_16+'+str(T16)+' E3_16+'+str(T16)+' E4_16+'+str(T16)+' E5_16+'+str(T16)+' E7_16+'+str(T16)+' E8_16+'+str(T17)+' E0_17+'+str(T17)+' E4_17+'+str(T17)+' E5_17+'+str(T17)+' E7_17+'+str(T17)+' E8_17+'+str(T17)+' E9_17+'+str(T18)+' E0_18+'+str(T18)+' E1_18+'+str(T18)+' E4_18+'+str(T18)+' E5_18+'+str(T18)+' E8_18+'+str(T18)+' E9_18+'+str(T19)+' E0_19+'+str(T19)+' E1_19+'+str(T19)+' E4_19+'+str(T19)+' E8_19+'+str(T19)+' E9_19+'+str(T20)+' E0_20+'+str(T20)+' E1_20+'+str(T20)+' E4_20+'+str(T20)+' E8_20+'+str(T20)+' E9_20+'+str(T21)+' E1_21+'+str(T21)+' E4_21+'+str(T21)+' E8_21+'+str(T21)+' E9_21+'+str(T22)+' E1_22+'+str(T22)+' E4_22+'+str(T22)+' E8_22+'+str(T22)+' E9_22+'+ str(T23)+' E4_23+'+str(T23)+' E8_23+'+str(T23)+' E9_23')
            f.write(';')
            f.write('\n')
            f.write('b ='+str(T3)+' B6_3+'+str(T4)+' B6_4+'+str(T5)+' B1_5+'+str(T5)+' B2_5+'+str(T5)+' B6_5+'+str(T6)+' B1_6+'+str(T6)+' B2_6+'+str(T6)+' B3_6+'+str(T6)+' B6_6+'+str(T6)+' B6_6+'+str(T6)+' B9_6+'+str(T7)+' B1_7+'+str(T7)+' B2_7+'+str(T7)+' B3_7+'+str(T7)+' B6_7+'+str(T7)+' B9_7+'+str(T8)+' B1_8+'+str(T8)+' B2_8+'+str(T8)+' B3_8+'+str(T8)+' B6_8+'+str(T8)+' B9_8+'+str(T9)+' B1_9+'+str(T9)+' B2_9+'+str(T9)+' B3_9+'+str(T9)+' B6_9+'+str(T9)+'B9_9+'+str(T10)+' B1_10+'+str(T10)+' B2_10+'+str(T10)+' B3_10+'+str(T10)+' B6_10+'+str(T10)+' B9_10+'+str(T11)+' B0_11+'+str(T11)+' B1_11+'+str(T11)+' B2_11+'+str(T11)+' B3_11+'+str(T11)+' B6_11+'+str(T11)+' B9_11+'+str(T12)+' B0_12+'+str(T12)+' B2_12+'+str(T12)+' B3_12+'+str(T12)+' B6_12+'+str(T13)+' B0_13+'+str(T13)+' B2_13+'+str(T13)+' B3_13+'+str(T13)+' B6_13+'+str(T13)+' B8_13+'+str(T14)+' B0_14+'+str(T14)+' B2_14+'+str(T14)+' B3_14+'+str(T14)+' B6_14+'+str(T14)+' B8_14+'+str(T15)+' B0_15+'+str(T15)+' B2_15+'+str(T15)+'B3_15+'+str(T15)+' B6_15+'+str(T15)+' B8_15+'+str(T16)+' B0_16+'+str(T16)+' B2_16+'+str(T16)+' B3_16+'+str(T16)+' B6_16+'+str(T16)+' B8_16+'+str(T17)+' B0_17+'+str(T17)+' B2_17+'+str(T17)+' B3_17+'+str(T17)+' B6_17+'+str(T17)+' B8_17+'+str(T18)+' B0_18+'+str(T18)+' B2_18+'+str(T18)+' B3_18+'+str(T18)+' B5_18+'+str(T18)+' B6_18+'+str(T19)+' B0_19+'+str(T19)+' B2_19+'+str(T19)+' B3_19+'+str(T19)+' B4_19+'+str(T19)+' B5_19+'+str(T19)+' B6_19+'+str(T20)+' B0_20+'+str(T20)+' B2_20+'+str(T20)+' B3_20+'+str(T20)+' B5_20+'+str(T20)+' B6_20+'+str(T21)+' B0_21+'+str(T21)+' B2_21+'+str(T21)+' B5_21+'+str(T21)+' B6_21+'+str(T21)+' B7_21+'+str(T22)+' B0_22+'+str(T22)+' B2_22+'+str(T22)+' B6_22+'+str(T22)+' B7_22+'+str(T23)+' B2_23+'+str(T23)+' B6_23+'+str(T23)+' B7_23')
            f.write(';')
            f.write('\n')
            f.write('a ='+str(T4)+' A6_4+'+str(T5)+' A6_5+'+str(T6)+' A4_6+'+str(T6)+' A6_6+'+str(T7)+' A4_7+'+str(T7)+' A6_7+'+str(T7)+' A8_7+'+str(T8)+' A4_8+'+str(T8)+' A6_8+'+str(T8)+' A8_8+'+str(T8)+' A9_8+'+str(T9)+' A4_9+'+str(T9)+' A6_9+'+str(T9)+' A8_9+'+str(T9)+' A9_9+'+str(T10)+' A4_10+'+str(T10)+' A6_10+'+str(T10)+' A8_10+'+str(T10)+' A9_10+'+str(T11)+' A4_11+'+str(T11)+' A8_11+'+str(T11)+' A9_11+'+str(T12)+' A3_12+'+str(T12)+' A4_12+'+str(T12)+' A7_12+'+str(T12)+' A8_12+'+str(T12)+' A9_12+'+str(T13)+' A3_13+'+str(T13)+' A7_13+'+str(T13)+' A8_13+'+str(T13)+' A9_13+'+str(T14)+' A3_14+'+str(T14)+' A7_14+'+str(T14)+' A8_14+'+str(T14)+' A9_14+'+str(T15)+' A3_15+'+str(T15)+' A7_15+'+str(T16)+' A3_16+'+str(T16)+' A7_16+'+str(T17)+' A3_17+'+str(T17)+' A7_17+'+str(T18)+' A1_18+'+str(T18)+' A3_18+'+str(T18)+' A5_18+'+str(T18)+' A7_18+'+str(T19)+' A1_19+'+str(T19)+' A2_19+'+str(T19)+' A3_19+'+str(T19)+' A5_19+'+str(T20)+' A0_20+'+str(T20)+' A1_20+'+str(T20)+' A2_20+'+str(T20)+' A3_20+'+str(T20)+' A5_20+'+str(T21)+' A0_21+'+str(T21)+' A1_21+'+str(T21)+' A2_21+'+str(T22)+' A0_22+'+str(T22)+' A1_22+'+str(T23)+' A0_23+'+str(T23)+' A1_23')
            f.write(';')
            f.write('\n')
            for x in range(10):
                for t in range(int(A2[x])-int(A1[x])+1):
                    if(t == int(A2[x])):
                        print(' ')
                    elif(t == 0):
                        print(' ')
                    else:
                        f.write('+')
                    f.write(('A' +str(x) +'_'+str(t+int(A1[x]))))
                f.write('=' + str(int(A[x]))+';')    
                f.write(('\n'))
            f.write('\n')

            for x in range(10):
                for t in range(int(B2[x])-int(B1[x])+1):
                    if(t == int(B2[x])):
                        print(' ')
                    elif(t == 0):
                        print(' ')
                    else:
                        f.write('+')
                    f.write(('B' +str(x) +'_'+str(t+int(B1[x]))))
                f.write('=' + str(int(B[x]))+';')
                f.write(('\n'))
            f.write('\n')         
            for x in range(10):
                for t in range(int(C2[x])-int(C1[x])+1):
                    if(t == int(C2[x])):
                        print(' ')
                    elif(t == 0):
                        print(' ')
                    else:
                        f.write('+')
                    f.write(('C' +str(x) +'_'+str(t+int(C1[x]))))
                f.write('=' + str(int(C[x]))+';')    
                f.write(('\n'))
            f.write('\n')
            for x in range(10):
                for t in range(int(D2[x])-int(D1[x])+1):
                    if(t == int(D2[x])):
                        print(' ')
                    elif(t == 0):
                        print(' ')
                    else:
                        f.write('+')
                    f.write(('D' +str(x) +'_'+str(t+int(D1[x]))))
                f.write('=' + str(int(D[x]))+';')    
                f.write(('\n'))
            f.write('\n')
            for x in range(10):
                for t in range(int(E2[x])-int(E1[x])+1):
                    if(t == int(E2[x])):
                        print(' ')
                    elif(t == 0):
                        print(' ')
                    else:
                        f.write('+')
                    f.write(('E' +str(x) +'_'+str(t+int(E1[x]))))
                f.write('=' + str(int(E[x]))+';')    
                f.write(('\n'))
            f.write('\n')     



            
            for x in range(10):
                for t in range(int(A2[x])-int(A1[x])+1):
                    f.write(('\n'))
                    f.write(('0 <= A' +str(x) +'_'+str(t+int(A1[x]))+'<='+str(1) + ';'))
                
                f.write(('\n'))
            for x in range(10):
                for t in range(int(B2[x])-int(B1[x])+1):
                    f.write(('\n'))
                    f.write(('0 <= B' +str(x) +'_'+str(t+int(B1[x]))+'<='+str(1) + ';'))
                f.write(('\n'))
                    
            for x in range(10):
                for t in range(int(C2[x])-int(C1[x])+1):
                    f.write(('\n'))
                    f.write(('0 <= C' +str(x) +'_'+str(t+int(C1[x]))+'<='+str(1) + ';'))
                f.write(('\n'))
            for x in range(10):
            
                for t in range(int(D2[x])-int(D1[x])+1):
                    f.write(('\n'))
                    f.write(('0 <= D' +str(x) +'_'+str(t+int(D1[x]))+'<='+str(1) + ';'))
                f.write(('\n'))
            for x in range(10):
           
                for t in range(int(E2[x])-int(E1[x])+1):
                    f.write(('\n'))
                    f.write(('0 <= E' +str(x) +'_'+str(t+int(E1[x]))+'<='+str(1) + ';'))
                f.write(('\n'))
               


            f.close()
            ABreak =[]
            BBreak =[]
            CBreak =[]
            DBreak =[]
            EBreak =[]
            lp = lpsolve('read_LP',fileName,NORMAL,'test model')
            lpsolve('solve',lp)
            print(lpsolve('get_objective',lp))
            f = lpsolve('get_variables', lp)[0]
            #print(lpsolve('get_constraints', lp))
            print(f)
            a = f[0]
            b = f[1]
            c = f[2]
            d = f[3]
            e = f[4]
            print(a,b,c,d,e)
            for j in range(24):
                  CBreak.append(f[j+5])
                  DBreak.append(f[j+28])
                  EBreak.append(f[j+56])
                  BBreak.append(f[j+84])
                  ABreak.append(f[j+112])

            print(ABreak)
            print('\n')
            print(BBreak )
            print('\n')
            print(CBreak )
            print('\n')
            print(DBreak )
            print('\n')
            print(EBreak )

              
            lpsolve('delete_lp',lp)


test1.to_csv('labelledData.csv',sep='\t')
    #z.close()


    #print(Task1)
    #print(Task2)
    #print(Task3)
    #print(Task4)
    #print(Task5)

