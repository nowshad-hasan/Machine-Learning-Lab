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


genderDict={'M':0,'F':1}
nationalityDict={'KW':0,'lebanon':1,'Egypt':2,'SaudiArabia':3,'USA':4,'Jordan':5,'venzuela':6,'Iran':7,
                'Tunis':8,'Morocco':9,'Syria':10,'Palestine':11,'Iraq':12,'Lybia':13}

birthPlaceDict={'KuwaIT':0,'lebanon':1,'Egypt':2,'SaudiArabia':3,'USA':4,'Jordan':5,'venzuela':6,'Iran':7,
                'Tunis':8,'Morocco':9,'Syria':10,'Palestine':11,'Iraq':12,'Lybia':13}

educationalStageDict={'lowerlevel':0,'MiddleSchool':1,'HighSchool':2}

gradeLevelDict={'G-01':0, 'G-02':1, 'G-03':2, 'G-04':3, 'G-05':4, 'G-06':5, 'G-07':6, 'G-08':7, 'G-09':8, 'G-10':9,'G-11':10,'G-12':11}

sectionIDict={'A':0,'B':1,'C':2}

courseDict={'English':0,'Spanish':1,'French':2,'Arabic':3,'IT':4,'Math':5,'Chemistry':6,'Biology':7,'Science':8,'History':9,'Quran':10,'Geology':11}

semesterDict={'F':0,'S':1}

parentResponsibleDict={'Mum':0,'Father':1}

parentAnsweringDict={'Yes':0,'No':1}

parentSchoolSatisfactionDict={'Good':0,'Bad':1}

studentAbsentDict={'Above-7':0,'Under-7':1}

classDict={'L':0,'M':1,'H':2}

featureList=[]
labelList=[]




with open("xAPI-Edu-Data.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    next(reader)
    # reader.next()
    # print sentences_file
    for row in reader:
        # print(row[0])
        #print(len(row))
        singleData=[]
        for i in range(0,len(row)-1):
            #print(i)
            if(i==0):
                singleData.append(genderDict[row[i]])
            elif(i==1):
                singleData.append(nationalityDict[row[i]])
            elif(i==2):
                singleData.append(birthPlaceDict[row[i]])
            elif(i==3):
                singleData.append(educationalStageDict[row[i]])
            elif(i==4):
                singleData.append(gradeLevelDict[row[i]])
            elif(i==5):
                singleData.append(sectionIDict[row[i]])
            elif(i==6):
                singleData.append(courseDict[row[i]])
            elif(i==7):
                singleData.append(semesterDict[row[i]])
            elif(i==8):
                singleData.append(parentResponsibleDict[row[i]])
            elif(i==9):
                singleData.append(int(row[i]))
            elif(i==10):
                singleData.append(int(row[i]))
            elif(i==11):
                singleData.append(int(row[i]))
            elif(i==12):
                singleData.append(int(row[i]))
            elif(i==13):
                singleData.append(parentAnsweringDict[row[i]])
            elif(i==14):
                singleData.append(parentSchoolSatisfactionDict[row[i]])
            elif(i==15):
                singleData.append(studentAbsentDict[row[i]])
        featureList.append(singleData);
        labelList.append(classDict[row[16]])

logreg_model = LogisticRegression(C=1e5)
linear_model=linear_model.LinearRegression()
svm_model=svm.SVC();
k_neighbor_model = neighbors.KNeighborsClassifier()
gaussian_model = GaussianNB()
decision_tree_model = tree.DecisionTreeClassifier()
MLP_classifier_model=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#scores = cross_val_score(linear_model, featureList, labelList, cv=5)

scoresLogreg_model = cross_val_score(logreg_model, featureList, labelList, cv=3)
scoresLinear_model = cross_val_score(linear_model, featureList, labelList, cv=3)
scoresSvm_model = cross_val_score(svm_model, featureList, labelList, cv=3)
scoresGaussian_model = cross_val_score(gaussian_model, featureList, labelList, cv=3)
scoresDecision_tree_model = cross_val_score(k_neighbor_model, featureList, labelList, cv=3)

predicted = cross_val_predict(linear_model, featureList, labelList, cv=10)
print(scoresLogreg_model," ",scoresLinear_model," ",scoresSvm_model," ",scoresGaussian_model," ",scoresDecision_tree_model);
# y=labelList
# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
#print(scores)
#print(len(predicted))
#print (len(featureList)," ",len(labelList))
#print featureList
#print gradeLevel['G-11']
