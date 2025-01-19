from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import seaborn as sns
import pandas as pd

#------------------------Part A---------------------------------------------------------------------------------------



class NeuralNetwork:

    def __init__(self, learning_rate, momentum, num_hidden_nodes):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.num_hidden_nodes=num_hidden_nodes
        

        #Binary classification
        self.num_output_nodes=1
    
    
        
        

    def test_loss(self, X_test, y_test):
        total_loss = 0
        total_samples = len(X_test)

        for i in range(total_samples):
            input_pattern = X_test[i]
            expected_output = y_test[i]
            output = self.forward(input_pattern)
            loss = 1/2 * (output - expected_output) ** 2
            total_loss += loss

        average_test_loss = total_loss / total_samples
        return average_test_loss

    def train_test_losses(self, X, y, X_train, y_train, epochs):

        self.iniWeights(X)


        train_loss=[]
        test_loss=[]
        f_tr_loss=0

        for i in range(epochs):

            
            tr_loss=self.fit(X, y, 1)
            ts_loss=self.test_loss(X_train, y_train)
            train_loss.append(tr_loss)
            test_loss.append(ts_loss)
            if(i==epochs-1):
                f_tr_loss=train_loss[i]
            
        
        
        return train_loss, test_loss, f_tr_loss


    

    def iniWeights(self, X):
        

        self.num_input_nodes=len(X[0])

        #Initialize input/hidden weights
        self.Input_weights=np.random.RandomState(2).randn(self.num_input_nodes, self.num_hidden_nodes)
        self.Hidden_weights=np.random.RandomState(2).randn(self.num_hidden_nodes, self.num_output_nodes)

        # Initialize momentum lists
        self.momentum_IW=np.zeros((self.num_input_nodes, self.num_hidden_nodes))
        self.momentum_HW=np.zeros((self.num_hidden_nodes, self.num_output_nodes))

        #Initialize biases
        self.hidden_bias= [0]*int(self.num_hidden_nodes)
        self.output_bias=0       
             


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)


    def forward(self, X):

        # Hidden layer
        hidden_input=np.dot(X, self.Input_weights)+ self.hidden_bias
        self.hidden=self.sigmoid(hidden_input)
        
        # Output layer (one node because binary classification)
        output_input=np.dot(self.hidden, self.Hidden_weights) +self.output_bias
        hidden_output=self.sigmoid(output_input)

        return hidden_output



    
    def backward(self, X, y, output):


        #calculate loss
        loss = 1/2*(output - y)**2

        #Calculate output error
        output_error=y-output

        #Output layer delta
        d_output=output_error*self.sigmoid_derivative(output)

        #Hidden layer deltas
        d_hidden=self.sigmoid_derivative(self.hidden)*np.dot(d_output, self.Hidden_weights.T)


        # Update Input weights and momentum array
        self.momentum_IW = self.learning_rate * np.outer(X, d_hidden) + self.momentum * self.momentum_IW
        self.Input_weights += self.momentum_IW

        # Update Hidden weights and momentum
        self.momentum_HW = self.learning_rate * np.outer(self.hidden, d_output) + self.momentum * self.momentum_HW
        self.Hidden_weights += self.momentum_HW

     
        #Update bias
        self.hidden_bias+=self.learning_rate*d_hidden
        self.output_bias+=self.learning_rate*d_output
        
        return loss

        
    
    def fit(self, X, y, epochs):

        

        total_loss=0
        for j in range(epochs):
            total_loss=0
            for i in range(len(X)):
                input_pattern = X[i]
                expected_output = y[i]
                output = self.forward(input_pattern)
                curr_loss=self.backward(input_pattern, expected_output, output)
                total_loss+=curr_loss
        return total_loss/len(X)
        
                
                
    
    def predict(self, X):
        return [1 if pred >= 0.5 else 0 for pred in self.forward(X)]

    

#------------------------Part B---------------------------------------------------------------------------------------


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0)



#------------------------B.1---------------------------------------------------------------------------------------


NN=NeuralNetwork(0.0001, 0, 256)
epochs=1400#(1400)

train_loss, test_loss, f_training_loss=NN.train_test_losses(X_train, y_train, X_test, y_test, epochs)

x = [i for i in range(1, epochs+1)]

plt.plot(x, train_loss, label='Average Train loss')
plt.plot(x, test_loss, label='Average Test loss')
plt.title("Average train/test loss")
plt.legend()
plt.show()


# sys.exit()



#------------------------B.2.1---------------------------------------------------------------------------------------

lr=[0.0001, 0.001, 0.01, 0.1, 1]
epochs=250#(250)
f_tr_loss=[]
acc=[]

for i in range(len(lr)):
    
    


    #1
    NN=NeuralNetwork(lr[i], 0, 32)

    train_loss, test_loss, f_training_loss=NN.train_test_losses(X_train, y_train, X_test, y_test, epochs)
    f_tr_loss.append(f_training_loss)

    x = [i for i in range(1, epochs+1)]

    plt.plot(x, train_loss, label='Average Train loss')
    plt.plot(x, test_loss, label='Average Test loss')
    plt.title("Average train/test loss")
    plt.legend()
    plt.show()


    #2
    print("Final training loss: ",f_training_loss)

    #3
    correct=0
    for l in range(len(X_test)):
        input=X_test[l]
        expected=y_test[l]
        prediction=NN.predict(input)
        if(prediction==expected):
            correct+=1
    accuracy=correct/len(X_test)
    print("Accuracy of test set: ", accuracy)
    acc.append(accuracy)

#4
plt.plot(lr, f_tr_loss)
plt.title("Final training loss for each learning rate ")
plt.legend()
plt.show()

#5
plt.plot(lr, acc)
plt.title("Accuracy for each learning rate ")
plt.legend()
plt.show()




# sys.exit()


#------------------------B.2.2---------------------------------------------------------------------------------------
hl=[8, 16, 32, 64, 128, 256, 512]
epochs=250#(250)
f_tr_loss=[]
acc=[]

for i in range(len(hl)):

    #6
    NN=NeuralNetwork(0.0001, 0, hl[i])

    train_loss, test_loss, f_training_loss=NN.train_test_losses(X_train, y_train, X_test, y_test, epochs)
    f_tr_loss.append(f_training_loss)

    

    x = [i for i in range(1, epochs+1)]

    plt.plot(x, train_loss, label='Average Train loss')
    plt.plot(x, test_loss, label='Average Test loss')
    plt.title("Average train/test loss")
    plt.legend()
    plt.show()


    #7
    print("Final training loss: ",f_training_loss)

    #8
    correct=0
    for l in range(len(X_test)):
        input=X_test[l]
        expected=y_test[l]
        prediction=NN.predict(input)
        if(prediction==expected):
            correct+=1
    accuracy=correct/len(X_test)
    print("Accuracy of test set: ", accuracy)
    acc.append(accuracy)
    

#9
plt.plot(hl, f_tr_loss)
plt.title("Final training loss for each number of hidden nodes ")
plt.legend()
plt.show()

#10
plt.plot(hl, acc)
plt.title("Accuracy for each number of hidden nodes")
plt.legend()
plt.show()


# sys.exit()




#------------------------B.2.3---------------------------------------------------------------------------------------
m=[0, 0.1, 0.25, 0.5, 1]
epochs=250#(250)
f_tr_loss=[]
acc=[]

for i in range(len(m)):

    #11
    NN=NeuralNetwork(0.001, m[i], 32)

    train_loss, test_loss, f_training_loss=NN.train_test_losses(X_train, y_train, X_test, y_test, epochs)
    f_tr_loss.append(f_training_loss)

    

    x = [i for i in range(1, epochs+1)]

    plt.plot(x, train_loss, label='Average Train loss')
    plt.plot(x, test_loss, label='Average Test loss')
    plt.title("Average train/test loss")
    plt.legend()
    plt.show()


    #12
    print("Final training loss: ",f_training_loss)

    #13
    correct=0
    for l in range(len(X_test)):
        input=X_test[l]
        expected=y_test[l]
        prediction=NN.predict(input)
        if(prediction==expected):
            correct+=1
    accuracy=correct/len(X_test)
    print("Accuracy of test set: ", accuracy)
    acc.append(accuracy)

#14
plt.plot(m, f_tr_loss)
plt.title("Final training loss for each momentum")
plt.legend()
plt.show()

#15
plt.plot(m, acc)
plt.title("Accuracy for each momentum")
plt.legend()
plt.show()



# sys.exit()


#------------------------B.2.4---------------------------------------------------------------------------------------
lr=[0.0001, 0.001, 0.01, 0.1, 1]
m=[0, 0.1, 0.25, 0.5, 1]
hl=[8, 16, 32, 64, 128, 256, 512]






f_tr_loss=[]
epochs=250#250
acc=[]
best_comb=[0,0,0]
best_acc=float('-inf')
best_train_loss=[]
best_test_loss=[]
best_f_training_loss=-1
best_accuracy=-1

for i in range(len(lr)):
    for j in range(len(m)):
        for k in range(len(hl)):

            NN=NeuralNetwork(lr[i], m[j], hl[k])
            
            

            train_loss, test_loss, f_training_loss=NN.train_test_losses(X_train, y_train, X_test, y_test, epochs)
            
            correct=0
            for l in range(len(X_test)):
                input=X_test[l]
                expected=y_test[l]
                prediction=NN.predict(input)
                if(prediction==expected):
                    correct+=1
            accuracy=correct/len(X_test)
            acc.append(accuracy)

            if(accuracy>best_acc):
                best_acc=accuracy
                best_comb=[lr[i],m[j],hl[k]]
                best_train_loss=train_loss.copy()
                best_test_loss=test_loss.copy()
                best_f_training_loss=f_training_loss
                best_accuracy=accuracy



print("Best combination based on accuracy: ", best_comb)


x = [i for i in range(1, epochs+1)]
plt.plot(x, best_train_loss, label='Average Train loss')
plt.plot(x, best_test_loss, label='Average Test loss')
plt.legend()
plt.show()

print("Final training loss of best combination: ",best_f_training_loss)
print("Accuracy of test set for the best combination: ", best_accuracy)



#Creating heatmap for accuracy of each combination

# Create a DataFrame to store the accuracies
df = pd.DataFrame(acc, columns=['Hidden Nodes'],index=pd.MultiIndex.from_product([lr, m, hl], names=['Learning Rate', 'Momentum', 'Hidden Layer Nodes']))


# Plot a heatmap of the accuracies
sns.heatmap(df.unstack(), annot=True, fmt=".2f")
plt.show()



















