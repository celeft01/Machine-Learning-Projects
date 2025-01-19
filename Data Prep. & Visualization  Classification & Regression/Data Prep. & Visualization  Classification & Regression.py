import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Importing csv files
adults=pd.read_csv("adults.csv")
adults_test=pd.read_csv("adults_test.csv")
boston=pd.read_csv("boston.csv")




#1
#Create new dataframe that only has males from the US
newDF=adults[adults['Sex']=='Male']
newDF2=newDF[newDF['Native Country']=='United-States']
print("Males in adults.csv: ", len(newDF2))

#2
#Create new dataframe with all adults with at least a masters, and check if everyone has a salary of more than 50k per year
min_bach=adults[adults['Education']=='Bachelors']

min_bach=pd.concat([min_bach, adults[adults['Education']=='Masters']], axis=0)
min_bach=pd.concat([min_bach, adults[adults['Education']=='Prof-school']], axis=0)
min_bach=pd.concat([min_bach, adults[adults['Education']=='Doctorate']], axis=0)


if('<=50K' in min_bach['Salary'].unique()):
    print("Adults with at least a Bachelors degree are not guaranteed to receive more than 50K per year")
else:
     print("Adults with at least a Bachelors degree are guaranteed to receive more than 50K per year")


#3
#Nested for loop to loop through all races and sex
race=adults['Race'].unique()
gender=adults['Sex'].unique()

for i in race:
    for j in gender:
        df1=adults[adults['Race']==i]
        df1=df1[df1['Sex']==j]
        print("Minimum, maximum, average and standard deviation of the hours-per week for pair ", i, j," :")
        print(df1['Hours Per Week'].min())
        print(df1['Hours Per Week'].max())
        print(df1['Hours Per Week'].mean())
        print(df1['Hours Per Week'].std())

#-------------------------------------------------------------------------------------------------------------------------

#1
print("adults.csv")
print("----------------------------------- BEFORE -----------------------------------")
print("Number of adults before: ",len(adults))
print(adults.isnull().sum())

adults = adults.dropna()
adults = adults.drop_duplicates()

print("----------------------------------- AFTER -----------------------------------")
print("Number of adults after: ",len(adults))
print(adults.isnull().sum())



print("\n\nadults_test.csv")
print("----------------------------------- BEFORE -----------------------------------")
print("Number of adults before: ",len(adults_test))
print(adults_test.isnull().sum())

adults_test = adults_test.dropna()
adults_test = adults_test.drop_duplicates()

print("----------------------------------- AFTER -----------------------------------")
print("Number of adults after: ",len(adults_test))
print(adults_test.isnull().sum())


#2 Ordinal encoding on Education
edu_levels = adults['Education'].unique()
print(edu_levels)
encoder = OrdinalEncoder(categories=[['Preschool','1st-4th','5th-6th','7th-8th','9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school','Doctorate']])
adults['education_encoded'] = encoder.fit_transform(adults[['Education']])
adults_test['education_encoded'] = encoder.fit_transform(adults_test[['Education']])

print("adults.csv")
print(adults[["Education", "education_encoded"]].head(20))
print("adults_test.csv")
print(adults_test[["Education", "education_encoded"]].head(20))


#3 Standardization
standard_scaler = StandardScaler()
adults["age_scaled"]= standard_scaler.fit_transform(adults[['Age']])
adults_test["age_scaled"] = standard_scaler.fit_transform(adults_test[['Age']])

print('adulst.csv')
print(adults[["Age", "age_scaled"]].head(20))
print('adults_test.csv')
print(adults_test[["Age", "age_scaled"]].head(20))


#4

#a
orio=0.003

cert_distribution = adults.groupby(['Native Country'])['Native Country'].count().to_frame('count').reset_index()


#Creating an 'Others' category
count1=cert_distribution['count'].sum()
cert_distribution.loc[cert_distribution['count']/count1<orio, 'Native Country']='Others'

fig = px.pie(cert_distribution, values='count',names='Native Country')
fig.show()






cert_distribution2 = adults_test.groupby(['Native Country'])['Native Country'].count().to_frame('count').reset_index()

#Creating an 'Others' category
count2=cert_distribution2['count'].sum()
cert_distribution2.loc[cert_distribution2['count']/count2<orio, 'Native Country']='Others'

fig2 = px.pie(cert_distribution2, values='count',names='Native Country')
fig2.show()




#b (the answer is education_encoded)


adults_new = adults[['Salary','Age', 'education_encoded','Hours Per Week']].copy()

adults_test_new = adults_test[['Salary','Age', 'education_encoded','Hours Per Week']].copy()




adults_new['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
adults_test_new['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)

sns.set(rc = {'figure.figsize':(12,8)})
sns.heatmap(adults_new.corr()[['Salary']].sort_values(by='Salary', ascending=False), vmin=-1,
                    vmax=1, annot=True, cmap='RdGy_r', xticklabels=True, yticklabels=True)
plt.show()

sns.set(rc = {'figure.figsize':(12,8)})
sns.heatmap(adults_test_new.corr()[['Salary']].sort_values(by='Salary', ascending=False), vmin=-1,
                    vmax=1, annot=True, cmap='RdGy_r', xticklabels=True, yticklabels=True)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------


#1
adults_reg=adults[['Salary','Age', 'education_encoded','Hours Per Week']].copy()
adults_test_reg=adults_test[['Salary','Age', 'education_encoded','Hours Per Week']].copy()

#Encoding salary
adults_reg['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
adults_test_reg['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)

#Create y and X for adults
y =adults_reg['Salary'].values
adults_reg.drop('Salary', axis=1, inplace=True)
X = []
for index, row in adults_reg.iterrows():
  X.append(row.values)

lr = LogisticRegression(max_iter=5000)
lr.fit(X,y)



#Create y1 and X1 for adults_test
y1 =adults_test_reg['Salary'].values

adults_test_reg.drop('Salary', axis=1, inplace=True)
X1 = []
for index, row in adults_test_reg.iterrows():
  X1.append(row.values)

prediction=lr.predict(X1)
print(prediction)


#2
matrix_lr = metrics.confusion_matrix(y1, prediction)
print("\nLR Confusion Matrix")
print("TN:",matrix_lr[0][0],"FP:",matrix_lr[0][1],"FN:",matrix_lr[1][0],"TP:",matrix_lr[1][1])
print(matrix_lr)
#TP means that the salary was predicted >50K and was >50K
#TN means that the salary was predicted <=50K and was <=50K
#FP means that the salary was predicted >50K but was <=50K
#FN means that the salary was predicted <=50K but was >50K


#3 
lr_acc = metrics.accuracy_score(y1,prediction)
lr_precision = metrics.precision_score(y1,prediction)
lr_recall = metrics.recall_score(y1,prediction)
lr_f1 = metrics.f1_score(y1,prediction)
print("Accuracy: ", lr_acc)
print("Precision: ", lr_precision)
print("Recall: ", lr_recall)
print("F1: ", lr_f1)

#4
lr_pred_proba = lr.predict_proba(X1)[::,1]
lr_fpr, lr_tpr, _ = metrics.roc_curve(y1,  lr_pred_proba)
lr_auc = metrics.roc_auc_score(y1, lr_pred_proba)
plt.plot(lr_fpr,lr_tpr,label="LR, auc="+str(lr_auc))
plt.legend(loc=4)
plt.show()

#---------------PART B----------------------------------------------------------------------------------------------------------


#1
#Bivariate feature
boston['RD'] = boston['RAD'] * boston['DIS']
print("\nBivariate Feature: RAD x tmdb_DIS")
print(boston[['RAD','DIS','RD']])

# Polynomial feature
boston['strong_RAD'] = boston['RAD'] ** 2
print("\nPolynomial feature RAD^2")
print(boston[['RAD','strong_RAD']])

# Custom feature
boston['DIS_to_RAD_ratio'] = boston['DIS'] / boston['RAD']
print("\nCustom feature: DIS to RAD ratio-> DIS/RAD")
print(boston[['DIS','RAD','DIS_to_RAD_ratio']])


#2

y = boston['MEDV'].values
boston.drop('MEDV', axis=1, inplace=True)
X = []
for index, row in boston.iterrows():
  X.append(row.values)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

#3
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
# print(lr_pred)

#4
lr_mse = mean_squared_error(y_test,lr_pred)
print("MSE of LR:",lr_mse)

#5
x_axis=[]
for i in range(len(y_test)):
   x_axis.append(i+1)
#print(x_axis)
plt.plot(x_axis, y_test, label="Real house prices")
plt.plot(x_axis, lr_pred, label="Predicted house prices")
plt.ylabel('Prices')
plt.show()