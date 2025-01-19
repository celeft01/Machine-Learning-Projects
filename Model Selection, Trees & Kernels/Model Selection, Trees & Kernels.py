import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

#Read csv files
adults=pd.read_csv("adults.csv")
adults_test=pd.read_csv("adults_test.csv")

#Drop na and dups
adults = adults.dropna()
adults = adults.drop_duplicates()
adults_test = adults_test.dropna()
adults_test = adults_test.drop_duplicates()

#Encoding Salary, Sex
adults['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
adults_test['Salary'].replace({'<=50K': 0, '>50K': 1}, inplace=True)

adults['Sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
adults_test['Sex'].replace({'Male': 0, 'Female': 1}, inplace=True)



#Ordinal encoding on Education
edu_levels = adults['Education'].unique()
encoder = OrdinalEncoder(categories=[['Preschool','1st-4th','5th-6th','7th-8th','9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school','Doctorate']])
adults['Education'] = encoder.fit_transform(adults[['Education']])
adults_test['Education'] = encoder.fit_transform(adults_test[['Education']])


#Removing extra columns
adults.drop(['Work Class','Marital Status','Occupation','Relationship','Race','Native Country'], axis=1, inplace=True)
adults_test.drop(['Work Class','Marital Status','Occupation','Relationship','Race','Native Country'], axis=1, inplace=True)



#PART A-----------------------------------------------------------------------------------------------------------------------------------
def finetune(clf, grid_param, rand_param, X, Y): 
    search_model = GridSearchCV(clf, grid_param,scoring='roc_auc', cv=2)
    search_model.fit(X,Y)
    grid_best=search_model.best_params_
    print("GridSearch - Best Hyperparameters:",grid_best)

    search_model2 = RandomizedSearchCV(estimator = clf,param_distributions = rand_param, scoring="roc_auc",cv = 2, n_iter=50)
    search_model2.fit(X,Y)
    rand_best=search_model2.best_params_
    print("RandomizedSearch - Best Hyperparameters:",rand_best)
    return grid_best , rand_best

def fit_and_evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    prediction=clf.predict(X_test)
    clf_pred_proba=clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  clf_pred_proba)
    auc = metrics.roc_auc_score(y_test, clf_pred_proba)
    return fpr, tpr, auc




#PART B-----------------------------------------------------------------------------------------------------------------------------------

#B1---------------------------------------------------------------------------------------------------------------------------------------

#Create y_train and X_train from adults
y_train =adults['Salary'].values
adults.drop('Salary', axis=1, inplace=True)
X_train = []
for index, row in adults.iterrows():
  X_train.append(row.values)


#Create y_test and X_test from adults_test
y_test =adults_test['Salary'].values
adults_test.drop('Salary', axis=1, inplace=True)
X_test = []
for index, row in adults_test.iterrows():
  X_test.append(row.values)





n_estimators = np.linspace(start=1, stop=50, num=50, dtype=int)
g_parameters = {'n_estimators':[5,10,25,50,100,150,200]}
r_parameters = {'n_estimators':n_estimators}








#Bagging Classifier
bdt = BaggingClassifier()
grid_best1, rand_best1=finetune(bdt, g_parameters, r_parameters, X_train, y_train)

fpra, tpra, auca=fit_and_evaluate(bdt, X_train, y_train, X_test, y_test)


bdt.set_params(**grid_best1)
fprb, tprb, aucb=fit_and_evaluate(bdt, X_train, y_train, X_test, y_test)


bdt.set_params(**rand_best1)
fprc, tprc, aucc=fit_and_evaluate(bdt, X_train, y_train, X_test, y_test)

plt.plot(fpra,tpra,label="Default, auc="+str(auca))
plt.plot(fprb,tprb,label="Grid best, auc="+str(aucb))
plt.plot(fprc,tprc,label="Rand best, auc="+str(aucc))
plt.title("Bagging Classifier")
plt.legend(loc=4)
plt.show()




#Random Forest Classifier
rf = RandomForestClassifier(random_state=0)
grid_best2, rand_best2=finetune(rf, g_parameters, r_parameters, X_train, y_train)

fpra, tpra, aucd=fit_and_evaluate(rf, X_train, y_train, X_test, y_test)


rf.set_params(**grid_best2)
fprb, tprb, auce=fit_and_evaluate(rf, X_train, y_train, X_test, y_test)


rf.set_params(**rand_best2)
fprc, tprc, aucf=fit_and_evaluate(rf, X_train, y_train, X_test, y_test)

plt.plot(fpra,tpra,label="Default, auc="+str(aucd))
plt.plot(fprb,tprb,label="Grid best, auc="+str(auce))
plt.plot(fprc,tprc,label="Rand best, auc="+str(aucf))
plt.title("Random Forest Classifier")
plt.legend(loc=4)

plt.show()



#XGB Classifier
xgb = XGBClassifier()
grid_best3, rand_best3=finetune(xgb, g_parameters, r_parameters, X_train, y_train)

fpra, tpra, aucg=fit_and_evaluate(xgb, X_train, y_train, X_test, y_test)


xgb.set_params(**grid_best3)
fprb, tprb, auch=fit_and_evaluate(xgb, X_train, y_train, X_test, y_test)


xgb.set_params(**rand_best3)
fprc, tprc, auci=fit_and_evaluate(xgb, X_train, y_train, X_test, y_test)

plt.plot(fpra,tpra,label="Default, auc="+str(aucg))
plt.plot(fprb,tprb,label="Grid best, auc="+str(auch))
plt.plot(fprc,tprc,label="Rand best, auc="+str(auci))
plt.title("XGB Classifier")
plt.legend(loc=4)
plt.show()



#B2---------------------------------------------------------------------------------------------------------------------------------------
best_bdt=''
best_rf=''
best_xgb=''
max_auc=max(auca, aucb, aucc)
if(max_auc==auca):
   best_bdt="default"
elif(max_auc==aucb):
   best_bdt="grid"
else:
   best_bdt="rand"

max_auc=max(aucd, auce, aucf)
if(max_auc==aucd):
   best_rf="default"
elif(max_auc==auce):
   best_rf="grid"
else:
   best_rf="rand"
  
max_auc=max(aucg, auch, auci)
if(max_auc==aucg):
   best_xgb="default"
elif(max_auc==auch):
   best_xgb="grid"
else:
   best_xgb="rand"

print("Best bdt: ",best_bdt," Best rf: ", best_rf," Best xgb: ",best_xgb)






# Time for bdt
bdt = BaggingClassifier()


bdt.set_params(**grid_best1)
start = time.time()
fit_and_evaluate(bdt, X_train, y_train, X_test, y_test)
end = time.time()
bdt_grid_time = end-start

bdt.set_params(**rand_best1)
start = time.time()
fit_and_evaluate(bdt, X_train, y_train, X_test, y_test)
end = time.time()
bdt_rand_time = end-start





# Time for rf
rf = RandomForestClassifier(random_state=0)


rf.set_params(**grid_best2)
start = time.time()
fit_and_evaluate(rf, X_train, y_train, X_test, y_test)
end = time.time()
rf_grid_time = end-start

rf.set_params(**rand_best2)
start = time.time()
fit_and_evaluate(rf, X_train, y_train, X_test, y_test)
end = time.time()
rf_rand_time = end-start





# Time for xgb
xgb = XGBClassifier()


xgb.set_params(**grid_best3)
start = time.time()
fit_and_evaluate(xgb, X_train, y_train, X_test, y_test)
end = time.time()
xgb_grid_time = end-start

xgb.set_params(**rand_best3)
start = time.time()
fit_and_evaluate(xgb, X_train, y_train, X_test, y_test)
end = time.time()
xgb_rand_time = end-start





# Time for svc
svc=SVC(probability=True)

start = time.time()
fit_and_evaluate(svc, X_train, y_train, X_test, y_test)
end = time.time()
svc_time = end-start


#Plotting times
x=['bdt_grid_time','bdt_rand_time','rf_grid_time','rf_rand_time','xgb_grid_time','xgb_rand_time','svc_time']
y=[bdt_grid_time,bdt_rand_time,rf_grid_time,rf_rand_time,xgb_grid_time,xgb_rand_time,svc_time]
plt.title("Times")
plt.bar(x, y)
plt.show()