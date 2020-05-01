# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from ast import literal_eval
import json
import re
'''
Depending on your choice of library you have to install that library using pip
'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

class NeuralNetwork():
    @staticmethod
    #note the self argument is missing i.e. why you have to search how to use static methods/functions
    def cross_entropy_loss(y_pred, y_true):
        '''implement cross_entropy loss error function here
        Hint: Numpy has a sum function already
        Numpy has also a log function
        Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
        after calculating -[y_true*log(y_pred)]'''
        return -(y_true * np.log(y_pred)).sum()
    @staticmethod
    def accuracy(y_pred, y_true):
        '''function to calculate accuracy of the two lists/arrays
        Accuracy = (number of same elements at same position in both arrays)/total length of any array
        Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
        matches = np.sum(y_pred == y_true)
        return matches

    
    @staticmethod
    def softmax(x):
        '''Implement the softmax function using numpy here
        Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
        Use keepdims=True for broadcasting
        You guys should have a pretty good idea what the size of returned value is.
        '''

        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    
    @staticmethod
    def sigmoid(x):
        '''Implement the sigmoid function using numpy here
        Sigmoid function is 1/(1+e^(-x))
        Numpy even has a exp function search for it.Eh?
        '''
        return 1 / (1 + np.exp(-x))
    
    def __init__(self,input_nodes, hidden_nodes, final_nodes,num_layers, nodes_per_layer):
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "num_layers" is the number of layers in your network 
        "input_shape" is the shape of the image you are feeding to the network
        "output_shape" is the number of probabilities you are expecting from your network'''

        self.nodes_per_layer = nodes_per_layer
        self.num_layers = num_layers # includes input layer
        self.hidden_shape = hidden_nodes
        self.input_shape = input_nodes
        self.output_shape = final_nodes
        self.__init_weights(nodes_per_layer)

    def __init_weights(self, nodes_per_layer):
        '''Initializes all weights and biases between -1 and 1 using numpy'''
        self.weights_ = []
        self.biases_ = []
        for i,_ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            
            weight_matrix = None
            self.weights_.append(weight_matrix)
            bias_vector = None
            self.biases_.append(bias_vector)
    
    def fit(self, Xs, Ys, epochs, lr=1e-3):
        '''Trains the model on the given dataset for "epoch" number of itterations with step size="lr". 
        Returns list containing loss for each epoch.'''
        history = []
       
        return history
    
    
    
    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)
        What is activation?
        In neural network you have inputs(x) and weights(w).
        What is first layer? It is your input right?
        A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
        A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
        Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
        '''

        activations = []






        self.input = input_data

        z_hidden_layer = input_data.dot(self.weights_[0]) + self.biases_[0]

        y_hidden_layer = self.sigmoid(z_hidden_layer)

        z_output_layer = y_hidden_layer.dot(self.weights_[1]) + self.biases_[1]

        y_output_layer = self.softmax(z_output_layer)

        activations = [y_hidden_layer, y_output_layer]


        
        return activations
    
    def backward_pass(self, targets, layer_activations):
        '''Executes the backpropogation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''
        deltas = []


        a1 = layer_activations[0]
        a2 = layer_activations[1]

        err2 = a2 - targets
        deriv2 = a2 * (1 - a2)
        delta_w2 = err2 * deriv2

        err1 = (self.weights_[1].dot(delta_w2.T)).T
        deriv1 = a1 * (1 - a1)
        delta_w1 = err1 * deriv1

        deltas = [delta_w1, delta_w2]

        return deltas
            
    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''

    x = layer_inputs[0]
    a1 = layer_inputs[1]

    w1 = self.weights_[0]
    w2 = self.weights_[1]

    b1 = self.biases_[0]
    b2 = self.biases_[1]

    delta_w1 = deltas[0]
    delta_w2 = deltas[1]

    w2 = w2 - lr * ((delta_w2.T.dot(a1)).T)
    b2 = b2 - lr * np.sum(delta_w2)  # unsure about this line

    w1 = w1 - lr * ((delta_w1.T.dot(x)).T)
    b1 = b1 - lr * np.sum(delta_w1)

    self.weights_[0] = w1
    self.weights_[1] = w2
        
    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        num_samples = Xs.shape[0]
        for i in range(num_samples):
            sample = Xs[i, :].reshape((1, self.input_shape))
            sample_prediction = self.forward_pass(sample)[-1]
            predictions.append(sample_prediction.reshape((self.output_shape,)))
        return np.array(predictions)
        # return predictions
    
    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        acc = self.accuracy(pred.argmax(axis=1), Ys.argmax(axis=1))
        loss = self.cross_entropy_loss(pred, Ys)
        return loss,acc
    def give_images(self,listDirImages):
        '''Returns the images and labels from the listDirImages list after reading
        Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
        in the provided folder. Similarly os.getcwd() returns you the current working
        directory. 
        For image reading use any library of your choice. Commonly used are opencv,pillow but
        you have to install them using pip
        "images" is list of numpy array of images 
        labels is a list of labels you read 
        '''
        images = []
        labels = []

        realDir = os.getcwd() + '/' + listDirImages
        for entry in os.listdir(realDir):
            # some might me files while other directories
            if "." not in entry:
                print('loading data for label===> ' + entry)
                # if it is not a file
                filesInDir = os.listdir(realDir + "/" + entry)
                for file in filesInDir:
                    image = Image.open(realDir + "/" + entry + "/" + file, 'r')
                    images.append(np.asarray(image))
                    labels.append(int(entry))
        z = list(zip(images, labels))
        random.shuffle(z)
        images, labels = zip(*z)
        return images, labels
        
        return images,labels
    def generate_labels(self,labels):
        '''Returns your labels into one hot encoding array
        labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
        Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
        Ex-> If label is 9 then one hot encoding shoudl be [0,0,0,0,0,0,0,0,0,1]
        Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
        "onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
        '''
        onehotlabels = []
        return onehotlabels
    def save_weights(self,fileName):
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
    def reassign_weights(self,fileName):
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
    def savePlot(self):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters pass to the savePlot function'''

def print_menu():
    print()
    print("MENU")
    print("Enter 1 to train")
    print("Enter 2 to predict")
    print("Enter 3 to exit")
    print()

def generate_inputarray(li):
    return np.array(li)


def generate_outputarray(f_name):
    return np.loadtxt(f_name,dtype=int)

def read_file(filename):
    a = open(filename).read().replace('\n', ' ')
    data_strings = re.findall('\[(.*?)\]', a)
    data = [[int(y) for y in x.strip().split()] for x in data_strings]
    return data



def main():
    np.random.seed(1)
    start = time.time()
    file_name = "train.txt"
    li = []  # List with all elements
    li2 = []
    start = time.time()


    li = read_file( file_name)
    print(li)

    input_nodes = 784
    hidden_nodes = 30
    final_nodes = 10
    num_layers = 3
    nodes_per_layer =  [784, 30, 10]
    learning_rate = 0.01
    epoch = 1
    nn = NeuralNetwork(input_nodes, hidden_nodes, final_nodes,num_layers, nodes_per_layer)


    # i_arr = generate_inputarray(li)  # test_num x input value
    # o_arr = generate_outputarray("train-labels.txt")  # 1 x output array
    #
    # result_file = "test.txt"
    # li2 = read_file(li2, result_file)
    # test_i_arr = generate_inputarray(li2)
    # test_o_arr = generate_outputarray("test-labels.txt")
    #
    # end = time.time()
    # print("Time for file read + array creation: " + str(end - start) + " seconds")
    #
    # start = time.time()
    #
    # col_size = i_arr.shape[1]
    # # nn.get_std_dev(i_arr)
    # end = time.time()
    # print("Time for std_dev + neural network creation: " + str(end - start) + " seconds")

    # user_input = "-1"
    # while (user_input != "3"):
    #     print_menu()
    #     user_input = input("Choice: ")
    #
    #     if user_input == "1":
    #         # train
    #         # nn.train(i_arr, o_arr)
    #         print("Epoch : " + str(epoch))
    #         epoch += 1
    #     if user_input == "2":
    #         # nn.guess(test_i_arr, test_o_arr)
    #         pass
main()

