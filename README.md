# Neural-Network
a versatile neural network written in python

uses numpy arrays to propogate through network

3 functions should be used: the constructor, forward, and train

the constructor takes 4 arguments (input_length,output_length,hidden_lengths,learning_rate)
hidden lengths is a list of integers which dictate how many hidden layers there will be and what the width of each layer is
learning rate is a float value indicating how fast the network will learn (I recommend between 0.001 and 0.0005)

forward takes a 2D numpy array with dimensions (K, input_length)
where K is the length of the set you wish to propogate and input_length is the same value as in the constructor

train takes 3 arguments (inputs, outputs, iterations)

inputs is a 2D numpy array with dimensions (K, input_length)
where K is the length of the set you wish to propogate and input_length is the same value as in the constructor

outputs is a 2D numpy array with dimensions (K, output_length)
where K is the length of the set you wish to propogate and output_length is the same value as in the constructor

