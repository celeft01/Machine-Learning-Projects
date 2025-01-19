from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import math
import random
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import time




#---------------------------------------Part A--------------------------------------------------------------------------------------------


X, y = make_blobs(n_samples=3000, centers=5, cluster_std=0.45, random_state=2)



#1
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xticks([])
plt.yticks([])
plt.show()

#Based on the scatter plot, I will select to create 5 clusters

#2
class MyKMeans:
    def __init__(self, clusters):
        self.clusters=clusters

    def fit(self, X):

        
        np.random.seed(2)
        random_indices=np.random.choice(X.shape[0], self.clusters, replace=False)
        self.centroids=X[random_indices]


        #Centroid for each point
        self.labels=[-1]*len(X)


        self.inertia=0

        




        while(True):
            old_centroids=copy.deepcopy(self.centroids)
            

            #Assigning data points to nearest centroid
            for i in range(len(X)):
                min_dist=float('inf')
                for j in range(len(self.centroids)):
                    distance = math.sqrt((self.centroids[j][0] - X[i][0])**2 + (self.centroids[j][1] - X[i][1])**2)#Distance
                    if(distance<min_dist):
                        min_dist=distance
                        self.labels[i]=j

            # Update centroids based on the mean of assigned data points
            for i in range(len(self.centroids)):
                sumX=0
                sumY=0
                count=0
                for j in range (len(X)):
                    if(self.labels[j]==i):
                        sumX+=X[j][0]
                        sumY+=X[j][1]
                        count+=1

                
                self.centroids[i][0]=sumX/count
                self.centroids[i][1]=sumY/count

            
            #Check for stoppage
            if np.array_equal(self.centroids, old_centroids):


                #Computing the inertia
                in_sum=0
                for i in range(len(self.centroids)):
                    for j in range (len(X)):
                        if(self.labels[j]==i):
                            in_sum+=((self.centroids[i][0] - X[j][0])**2 + (self.centroids[i][1] - X[j][1])**2)#Squared Eucledian distance\
                    
                self.inertia=in_sum
                

                break


    def predict(self, X):
        labels = [0]*len(X)
        for i in range(len(X)):
            min_dist=float('inf')
            for j in range(len(self.centroids)):
                distance = math.sqrt((self.centroids[j][0] - X[i][0])**2 + (self.centroids[j][1] - X[i][1])**2)#Eucledian distance
                if(distance<min_dist):
                    min_dist=distance
                    labels[i]=j
        return labels





#3
scores = {}
my_scores={}

for k in range (2,16):

    kmeans = KMeans(n_clusters=k, init='random', random_state=2).fit(X)
    scores[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')

    my_kmeans = MyKMeans(k)
    my_kmeans.fit(X)
    my_scores[k] = silhouette_score(X, my_kmeans.labels, metric='euclidean')



plt.figure(figsize=(20, 10))
plt.plot(list(scores.keys()), list(scores.values()),marker='o', label="sklearn")
plt.plot(list(my_scores.keys()), list(my_scores.values()),marker='o', label="my_kmeans")
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Co")
plt.grid()
plt.legend()
plt.show()





#4
scores = {}
my_scores={}
for k in range (2,16):

    kmeans = KMeans(n_clusters=k, init='random', random_state=2).fit(X)
    scores[k] = kmeans.inertia_

    my_kmeans = MyKMeans(k)
    my_kmeans.fit(X)
    my_scores[k]=my_kmeans.inertia

print(scores)
print(my_scores)

# Plotting the Inertia Scores to create the "Elbow Method Chart"
plt.figure(figsize=(20, 10))
plt.plot(list(scores.keys()), list(scores.values()),marker='o', label='sklearn')
plt.plot(list(my_scores.keys()), list(my_scores.values()),marker='o', label='my_kmeans')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Distances (samples->closest cluster center)")
plt.grid()
plt.legend()
plt.show()





#5

#a
m=MyKMeans(5)
m.fit(X)
centroids_array = np.array(m.centroids)

plt.figure(figsize=(10, 10))
colors = np.array(["#177EC8", "#FFAB00", "#3DAFA5", "#F123F1", "#A5151E", "#984EA4",
                   "#000000", "#ABCDEF", "#123ABC", "#ABC123"])
plt.scatter(X[:, 0], X[:, 1], s=10, c=colors[m.labels])  # Use 'c' instead of 'color'
plt.title("K-Means")
plt.xticks([])
plt.yticks([])
plt.show()

#b
m=MyKMeans(5)
m.fit(X)
centroids_array = np.array(m.centroids)

plt.figure(figsize=(10, 10))
colors = np.array(["#177EC8", "#FFAB00", "#3DAFA5", "#F123F1", "#A5151E", "#984EA4",
                   "#000000", "#ABCDEF", "#123ABC", "#ABC123"])
plt.scatter(X[:, 0], X[:, 1], s=10, c=colors[m.labels])  # Use 'c' instead of 'color'
plt.scatter(centroids_array[:, 0], centroids_array[:, 1],
            s=350, marker='o', c='red', edgecolor='black', label='centroids')
plt.title("K-Means")
plt.xticks([])
plt.yticks([])
plt.show()

#---------------------------------------Part B-------------------------------------------------------------------------------------------------

#Reading csv files
tr = pd.read_csv("transactions_mini.csv")
tr_val = pd.read_csv("transactions_mini_validation.csv")

#Getting the class column and removing it from the datasets
class_tr=tr['Class'].values
tr.drop('Class', axis=1, inplace=True)
class_tr_val=tr_val['Class'].values
tr_val.drop('Class', axis=1, inplace=True)


print(tr)

#2
#Legit and fraudulent data in original dataset
legit=np.count_nonzero(class_tr == 0)
fraud=np.count_nonzero(class_tr == 1)

print("Fraudulent transactions in original dataset:", fraud)
print("Legitimate transactions in original dataset:", legit)

#3
#Percentage of fraudulent transactions
percentage=(fraud/len(class_tr))*100
print("Percentage of fraudulent transactions in original dataset:", percentage)



# We estimate how many outliers in the dataset (1.68%) and set that as the contamination parameter.
outliers_fraction = 0.0168




#Initializing and training Isolation Forest
iso_forest = IsolationForest(contamination=outliers_fraction, random_state=0)

start=time.time()
iso_forest.fit(tr)
end=time.time()


prediction_iso_forest = iso_forest.predict(tr)

iso_time=end-start




#Initializing and training One Class SVM
oc_svm = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1) # nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1].

start=time.time()
oc_svm.fit(tr)
end=time.time()


prediction_oc_svm=oc_svm.predict(tr)


oc_time=end-start


print(prediction_iso_forest)
print(prediction_oc_svm)

#4
fraud_iso_forest=np.count_nonzero(prediction_iso_forest == -1)
fraud_oc_svm=np.count_nonzero(prediction_oc_svm == -1)
print("Fraudulent transactions captured by IsolationForest:", fraud_iso_forest)
print("Fraudulent transactions captured by OneClassSVM:", fraud_oc_svm)


#Counting correctly and incorrectly predicted fraudulent transactions
incorrect_iso=0
correct_iso=0
incorrect_oc=0
correct_oc=0

for i in range(len(prediction_oc_svm)):
    if(prediction_iso_forest[i]==-1 and class_tr[i]==1):
        correct_iso+=1
    elif(prediction_iso_forest[i]==-1 and class_tr[i]==0):
        incorrect_iso+=1

    if(prediction_oc_svm[i]==-1 and class_tr[i]==1):
        correct_oc+=1
    elif(prediction_oc_svm[i]==-1 and class_tr[i]==0):
        incorrect_oc+=1

#5
print("Legitimate transactions incorrectly classified as fraudulent by IsolationForest:", incorrect_iso)
print("Legitimate transactions incorrectly classified as fraudulent by OneClassSVM:", incorrect_oc)

#6 
percentage1=(correct_iso/fraud)*100
percentage2=(correct_oc/fraud)*100
print("Percentage of frauds in the original dataset that were detected by IsolationForest:", percentage1)
print("Percentage of frauds in the original dataset that were detected by OneClassSVM:", percentage2)

#7
print("Time to train IsolationForest:", iso_time)
print("Time to train OneClassSVM:", oc_time)



#10
iso_val_pred=iso_forest.predict(tr_val)
oc_val_pred=oc_svm.predict(tr_val)

fraud1=np.count_nonzero(iso_val_pred == -1)
fraud2=np.count_nonzero(oc_val_pred == -1)

print("Number of anomalies IsolationForest detected:", fraud1)
print("Number of anomalies OneClassSVM detected:", fraud2)