from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
import csv
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt


schoolDict={'GP':0,'MS':1}
sexDict={'F':0,'M':1}
addressDict={'U':0,'R':1}
familySizeDict={'LE3':0,'GT3':1}
parentStatusDict={'T':0,'A':1}
motherJobDict={'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
fatherJobDict={'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
reasonDict={'home':0,'reputation':1,'course':2,'other':3}
guardianDict={'mother':0,'father':1,'other':2}
schoolSupportDict={'yes':0,'no':1}
familySupportDict={'yes':0,'no':1}
paidDict={'yes':0,'no':1}
extraActivityDict={'yes':0,'no':1}
nurseryDict={'yes':0,'no':1}
higherDict={'yes':0,'no':1}
internetDict={'yes':0,'no':1}
romanticDict={'yes':0,'no':1}

featureList=[]
labelList=[]


with open("student-mat.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=';')
    # reader.next()
    next(reader)
    # print sentences_file
    #print(reader);
    for row in reader:
        #print(row.split(";"))
        singleData=[]
        for i in range(0,len(row)-1):
            #print(i)
            if(i==0):
                singleData.append(schoolDict[row[i]])
            elif(i==1):
                singleData.append(sexDict[row[i]])
            elif(i==2):
                singleData.append(int(row[i]))
            elif(i==3):
                singleData.append(addressDict[row[i]])
            elif(i==4):
                singleData.append(familySizeDict[row[i]])
            elif(i==5):
                singleData.append(parentStatusDict[row[i]])
            elif(i==6):
                singleData.append(int(row[i]))
            elif(i==7):
                singleData.append(int(row[i]))
            elif(i==8):
                singleData.append(motherJobDict[row[i]])
            elif(i==9):
                singleData.append(fatherJobDict[row[i]])
            elif(i==10):
                singleData.append(reasonDict[row[i]])
            elif(i==11):
                singleData.append(guardianDict[row[i]])
            elif(i==12):
                singleData.append(int(row[i]))
            elif(i==13):
                singleData.append(int(row[i]))
            elif(i==14):
                singleData.append(int(row[i]))
            elif(i==15):
                singleData.append(schoolSupportDict[row[i]])
            elif(i==16):
                singleData.append(familySupportDict[row[i]])
            elif(i==17):
                singleData.append(paidDict[row[i]])
            elif(i==18):
                singleData.append(extraActivityDict[row[i]])
            elif(i==19):
                singleData.append(nurseryDict[row[i]])
            elif(i==20):
                singleData.append(higherDict[row[i]])
            elif(i==21):
                singleData.append(internetDict[row[i]])
            elif(i==22):
                singleData.append(romanticDict[row[i]])
            elif(i==23):
                singleData.append(int(row[i]))
            elif(i==24):
                singleData.append(int(row[i]))
            elif(i==25):
                singleData.append(int(row[i]))
            elif(i==26):
                singleData.append(int(row[i]))
            elif(i==27):
                singleData.append(int(row[i]))
            elif(i==28):
                singleData.append(int(row[i]))
            elif(i==29):
                singleData.append(int(row[i]))
            elif(i==30):
                singleData.append(int(row[i]))
            elif(i==31):
                singleData.append(int(row[i]))
        featureList.append(singleData);
        #print(row[0])
        labelList.append(int(row[32]))
#print(featureList)
logreg_model = LogisticRegression(C=1e5)
linear_model=linear_model.LinearRegression()
svm_model=svm.SVC();
k_neighbor_model = neighbors.KNeighborsClassifier()
gaussian_model = GaussianNB()
decision_tree_model = tree.DecisionTreeClassifier()
MLP_classifier_model=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2, 1), random_state=1)
#MLP_classifier_model.fit(featureList,labelList);
#scoresNeuralNetwork_model=cross_val_score(MLP_classifier_model, featureList, labelList, cv=3)
scoresLogreg_model = cross_val_score(logreg_model, featureList, labelList, cv=3)
scoresLinear_model = cross_val_score(linear_model, featureList, labelList, cv=3)
scoresSvm_model = cross_val_score(svm_model, featureList, labelList, cv=3)
scoresGaussian_model = cross_val_score(gaussian_model, featureList, labelList, cv=3)
scoresDecision_tree_model = cross_val_score(k_neighbor_model, featureList, labelList, cv=3)
#print(MLP_classifier_model.score)

predicted = cross_val_predict(linear_model, featureList, labelList, cv=10)
print(scoresLogreg_model," ",scoresLinear_model," ",scoresSvm_model," ",scoresGaussian_model," ",scoresDecision_tree_model);
