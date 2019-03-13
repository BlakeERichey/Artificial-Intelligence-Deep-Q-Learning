##Ai using Deep Q Learning to learn how to drive a car
##Used in map.py

#---------------Imports---------------
import numpy as np    #useful for performing operations with arrays
import random         #used for experience replay
import os             #used to save and load trained NN
import torch          #Neural network API
import torch.nn as nn #neural netork
import torch.nn.functional as F       #performs loss function calculations
import torch.optim as optim           #optimizer for stochastic gradient descent
import torch.autograd as autograd     #used to get Variable
from   torch.autograd import Variable #used to contain tensor and gradient


#---------------Neural Network---------------
class Network(nn.Module):
  
  def __init__(self, inputQty, numActions):
    super(Network, self).__init__() #run parent constructor
    self.inputQty   = inputQty      #inputQty:   number of input neurons
    self.numActions = numActions    #numActions: number of possible actions for AI to take
    
    #Connections
    #full connection 1, connection between input layer and hidden layer
    #full connection 2, connection between hidden layer and output layer
    self.fc1 = nn.Linear(inputQty, 30)   #30 neurons in 1st hidden layer
    self.fc2 = nn.Linear(30, numActions) #30 neurons in hidden layer, numActions possible decisions

  def forward(self, state):
    layer1  = F.relu(self.fc1(state)) #relu = rectifier activation function
    qValues = self.fc2(layer1)        #feed activation of first hidden layer to ouutput layer
    return qValues  

#---------------Experience Replay---------------
class Replay(object):

  #capacity: variable to determine ER sample qty
  #memory:   last n events, controlled by push function
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory   = []

  #event: tuple of 4 elements (last state, new state, last action, last reward)
  #adds events to self.memory
  def push(self, event):
    self.memory.append(event)
    self.helper_constrain()  
  
  #remove elements from beginning until no more than capacity number of events exist
  def helper_constrain(self):
    if(len(self.memory) > self.capacity):
      del self.memory[0]      #remove first element
      self.helper_constrain()
    else:
      return 

  def getSample(self, batchSize):
    #convert ((ls, ns, la, lr), ...) => ((ls1, ls2...), (ns1, ns2...) ...)
    samples = zip(*random.sample(self.memory, batchSize))
    #convert samples to Variables containing Tensors and Gradients
    return map(lambda x: Variable(torch.cat(x)), samples)

#---------------Deep Q Learning---------------
