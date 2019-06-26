import numpy as np
import pandas as pd

#https://www.kaggle.com/uciml/pima-indians-diabetes-database
df = pd.read_csv('diabetes.csv')
df.info()

print("Nulls")
print("=====")
print(df.isnull().sum())

print("0s")
print("==")
print(df.eq(0).sum())

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
    'Age']] = \
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
    'Age']].replace(0,np.NaN)
df.fillna(df.mean(), inplace = True) # replace NaN with the mean
print(df.eq(0).sum())

corr = df.corr()
print(corr)

#------------------------------------------------------------------------------#
#Visualizes the corr
%matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)

ax.set_xticklabels(df.columns)
plt.xticks(rotation = 90)

ax.set_yticklabels(df.columns)
plt.yticks(ticks)

for i in range(df.shape[1]):
    for j in range(9):
        text = ax.text(j, i, round(corr.iloc[i][j],2), ha="center", va="center", color="w")
plt.show()

#------------------------------------------------------------------------------#
#Visualizes the corr using seaborn
import seaborn as sns

sns.heatmap(df.corr(), annot=True)
fig = plt.gcf()
fig.set_size_inches(8,8)

print(df.corr().nlargest(4, 'Outcome').index)
print(df.corr().nlargest(4, 'Outcome').values[:, 8])

#------------------------------------------------------------------------------#
#Use linear kernel SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score

X = df[['Glucose', 'BMI', 'Age']]
y = df.iloc[:, 8]

linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y, cv=10, scoring='accuracy').mean()
print(linear_svm_score)

result = []
result.append(linear_svm_score)

#------------------------------------------------------------------------------#
#Use RBF kernel SVM
rbf = svm.SVC(kernel='rbf', gamma='auto')
rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
print(rbf_score)
result.append(rbf_score)

#------------------------------------------------------------------------------#
#Print accuracy results
algorithms = ["SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result, index = algorithms)
cv_mean.columns = ["Accuracy"]
cv_mean.sort_values(by="Accuracy",ascending=False)

#------------------------------------------------------------------------------#
#Training and Saving the Model
lsvm = svm.SVC(kernel='linear').fit(X, y)

import pickle
filename = 'diabetes.sav' #save the model to disk
pickle.dump(lsvm, open(filename, 'wb')) #write to file using write and binary mode
loaded_model = pickle.load(open(filename, 'rb')) #load the model

Glucose = 65
BMI = 70
Age = 50

prediction = loaded_model.predict([[Glucose, BMI, Age]])
print(prediction)
if (prediction[0]==0):
    print("Non-diabetic")
else:
    print("Diabetic")

#proba = loaded_model.predict_proba([[Glucose, BMI, Age]])
#print(proba)
#print("Confidence: " + str(round(np.amax(proba[0]) * 100, 2)) + "%")

#------------------------------------------------------------------------------#
#Deploying the Model


