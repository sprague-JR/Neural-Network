import numpy as np
import random
class Neural_Network(object):
    def __init__(self,input_length,output_length,hidden_lengths, learning_rate):
    #parameters
        self.inputSize = input_length
        self.outputSize = output_length
        
        self.delta = learning_rate
        #initialise with random weights
        W1 = np.random.randn(self.inputSize, hidden_lengths[0])
        #weight matrix from input to hidden layer
        
        W_hidden = [np.random.randn(hidden_lengths[i],hidden_lengths[i+1]) for i in range(len(hidden_lengths) - 1)]
        #weight matrix from each layer in hidden layer to the next
        
        W2 = np.random.randn(hidden_lengths[-1], self.outputSize) 
        #weight matrix from hidden to output layer
        
        self.weight_matrices = [W1] + W_hidden + [W2]
        
        
    def forward(self, X):
        #forward propagation through the network
        self.propogation_data = []
        temp = X
        self.propogation_data.append([temp,self.sigmoid(temp)])

        for i in self.weight_matrices:
            temp = np.dot(self.propogation_data[-1][1],i)
            self.propogation_data.append([temp,self.sigmoid(temp)])
            
            #uses dot product between layers to act as propogation between layers
            #uses sigmoid as activation function
        return self.propogation_data[-1][1]
        
        
        

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))
    
    def sigmoid_prime(self,s):
        #derivitive of activation function
        return self.sigmoid(s) * (1 - self.sigmoid(s))
    
    def error(self,outputs,reals):
        returner = reals - outputs
        return returner
    
    def backward(self, X, y, o):
        # backward propgate through the network
        
        error_delta = []
        
        error_delta.append((self.error(o,y),self.error(o,y) * self.sigmoid_prime(o)))
        iterator = len(self.weight_matrices)
        
        for i in range(iterator -1,0,-1):
            temp_error = error_delta[-1][1].dot(self.weight_matrices[i].T)
            temp_delta = temp_error * self.sigmoid_prime(self.propogation_data[i][1])
        
            error_delta.append((temp_error,temp_delta))
        
        
        error_delta = error_delta[::-1]
        #invert error_deltas to align them with appropriate weight matrix
        
        for i in range(len(self.weight_matrices)):

            temp = self.propogation_data[i][1].T.dot(error_delta[i][1]) * self.delta
            self.weight_matrices[i] = self.weight_matrices[i] + temp
            #adjust weight matrices by appropriate values

        
    
    def train(self, X,y, iterations):
        temp_input = X
        temp_output = y
        temp_input = np.array(temp_input)
        temp_output = np.array(temp_output)
        o = 0
        for i in range(iterations):
            o = self.forward(temp_input)
            print("error on pass: " + str(i) + " is " + str(self.get_loss(temp_output,o)))
            self.backward(temp_input,temp_output,o)
            
            #shuffle the training set each time it is used to prevent over learning
            shuffle = random.sample(range(len(X)),len(X))
            temp_input = [X[i] for i in shuffle]
            temp_output = [y[i] for i in shuffle]
            temp_input = np.array(temp_input)
            temp_output = np.array(temp_output)

    def get_loss(self,X,y):
        #calculate the average error in the training data
        #only used as a debug to ensure improvement
        return np.mean(np.square(X - y))
