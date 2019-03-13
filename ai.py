##Ai using Deep Q Learning to learn how to drive a car
##Used in map.py

#Imports
import numpy as np    #useful for performing operations with arrays
import random         #used for experience replay
import os             #used to save and load trained NN
import torch          #Neural network API
import torch.nn as nn #neural netork
import torch.nn.functional as F       #performs loss function calculations
import torch.optim as optim           #optimizer for stochastic gradient descent
import torch.autograd as autograd     #used to get Variable
from   torch.autograd import Variable #used to contain tensor and gradient



class Network(nn.Module):
  
  def __init__(self, inputQty, numActions):
    super(Network, self).__init__() #run parent constructor
    self.inputQty   = inputQty      #inputQty:   number of input neurons
    self.numActions = numActions    #numActions: number of possible actions for AI to take
    
    #Connections
    #full connection 1, connection between input layer and hidden layer
    #full connection 2, connection between hidden layer and output layer
    self.fc1 = nn.Linear(inputQty, 30) #30 neurons in 1st hidden layer
    self.fc2 = nn.Linear(30, numActions) #30 neurons in hidden layer, numActions possible decisions

  def forward(self, state):
    layer1 = F.relu(self.fc1(state)) #relu = rectifier activation function
    qValues = self.fc2(layer1)       #feed activation of first hidden layer to ouutput layer
    return qValues  
