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

#---------------Deep Q Network--------------- Step 9
class Dqn():

  #inputQty:   number of inputs in NN input layer
  #numActions: number of outputs in NN output layer
  #gamma:     delay coefficient
  def __init__(self, inputQty, numActions, gamma):
    self.lastAction   = 0
    self.lastReward   = 0
    self.rewardWindow = []      #mean of rewards over time
    self.gamma        = gamma
    self.memory       = Replay(100000) #capacity of memory
    self.model        = Network(inputQty, numActions)
    self.lastState    = torch.Tensor(inputQty).unsqueeze(0)
    self.optimizer    = optim.Adam(self.model.parameters(), lr = 0.001)

  def selectAction(self, state):  #Step 10
    #values of probabilities of q values
    probs  = F.softmax(self.model(Variable(state, volatile = True)) * 0) #T = 7, inflates q values so q network is more confident in its action
    action = probs.multinomial() #decide action
    return action.data[0,0] 

  def learn(self, batchState, batchNextState, batchReward, batchAction): #step11
    outputs = self.model(batchState).gather(1, batchAction.unsqueeze(1)).squeeze(1) #1 is action
    nextOutputs = self.model(batchNextState).detach().max(1)[0] #take all the batches of states seperate them into tuples, find max of the actions, out of state
    target = self.gamma * nextOutputs + batchReward #guess
    tdLoss = F.smooth_l1_loss(outputs, target)      #guess - actual
    self.optimizer.zero_grad()  #reset optimizer
    tdLoss.backward(retain_variables = True)  #back propogate tsLoss through network
    self.optimizer.step() #update weights

  #update states after an action has been done
  def update(self, reward, signal):
    newState = torch.Tensor(signal).float().unsqueeze(0) #convert state value sent by car readings to a tensor
    self.memory.push((self.lastState, newState, torch.LongTensor([int(self.lastAction)]), torch.Tensor([self.lastReward]))) #add state results to memory
    action = self.selectAction(newState) #determine new action to take
    if( len(self.memory.memory) > 100 ):  #when batch size is enough, learn
      batchState, batchNextState, batchReward, batchAction = self.memory.getSample(100)
      self.learn(batchState, batchNextState, batchReward, batchAction)
    self.lastAction = action
    self.lastState  = newState
    self.lastReward = reward 
    self.rewardWindow.append(reward)
    if( len(self.rewardWindow) > 1000 ):
      del self.rewardWindow[0]
    return action

  def score(self):
    return sum(self.rewardWindow) / ( len(self.rewardWindow) + 1 )

  def save(self):
    torch.save(
      {
        'state_dict': self.model.state_dict(),
        'optimizer' : self.optimizer.state_dict()
      },
      'last_brain.pth'  #file name
    )

  def load(self):
    if(os.path.isfile('last_brain.pth')):
      print('=> loading checkpoint...')
      checkpoint = torch.load('last_brain.pth')
      self.model.load_state_dict(checkpoint['state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])
      print('done.')
    else:
      print('No checkpoint found...')