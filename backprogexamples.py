
import math
import random

# Neural networks
#
# Levels with neurons are numbered 1..lastLevel.
# Input nodes are level 0.
# Inputs are numbered 0..nOfInputs-1
# The rest of the nodes ("neurons") are nOfInputs..lastNeuron
# There is special node -1 with activation 1, which is added as an implicit
# predecessor to every neuron node (the 'bias' input).
#
# firstNeuron      : index of first neuron
# lastNeuron       : index of last neuron
# nodesOfLevel(l)  : return list of nodes (inputs or neurons) of level l
# setWeight(i,j,w) : set weight of (i,j) to w where i,j are inputs/neurons
# getWeight(i,j)   : get weight of (i,j) where i,j are inputs/neurons
# incoming(j)      : list of predecessors i of node j (connection i,j)
# outgoing(i)      : list of successors j of node i (connection i,j)
# level[i]         : level of node i
#
# NeuralNetwork(n,l,r0,r1) creates a new neural network with
#    n inputs (with inputs numbered from 0 to n-1)
#    l = [l1,...,lm] where l1,...,lm are lists of neuron indices,
#       with lowest neuron index equalling n)
#    [r0,r1] range of initial random weights for connections
#

class NeuralNetwork:

  def __init__(self,nOfInputs,levels,randLo=0,randHi=1):
    self.firstInput = 0
    self.lastInput = nOfInputs-1

    allNodes = list(set().union(*levels))
    
    self.firstNeuron = min(allNodes)
    self.lastNeuron = max(allNodes)

    self.firstLevel = 0
    self.lastLevel = len(levels)

    self.levels = [list(range(0,nOfInputs))] + levels

    self.outputs = levels[-1]

    self.predecessors = dict()
    self.successors = dict()
    self.level = dict()

    self.w = dict()

    # all inputs are at level 0
    for i in range(0,nOfInputs):
      self.level[i] = 0
    # other nodes are on higher levels
    for i in allNodes:
      for l in range(0,len(levels)+1):
        if i in self.levels[l]:
          self.level[i] = l

    for i in range(self.firstNeuron,self.lastNeuron+1):
      if(self.level[i] == 0):
         self.predecessors[i] = []
      else:
         self.predecessors[i] = [-1] + self.levels[self.level[i]-1]
      if(self.level[i] == self.lastLevel):
         self.successors[i] = []
      else:
         self.successors[i] = self.levels[self.level[i]+1]
    self.successors[-1] = allNodes
    self.predecessors[-1] = []
    
    for j in range(self.firstNeuron,self.lastNeuron+1):
      for i in self.predecessors[j]:
        self.w[i,j] = random.uniform(randLo,randHi)

  def nodesOfLevel(self,level):
    return self.levels[level]

  def getWeight(self,node1,node2):
    return self.w[node1,node2]

  def setWeight(self,node1,node2,w):
    self.w[node1,node2] = w

  def incoming(self,node):
    return self.predecessors[node]

  def outgoing(self,node):
    return self.successors[node]

# The logistic function for sigmoid neurons

def g(inn):
  return 1/(1+math.exp(-inn));

# Calculate the activations of all neurons

def evaluate(nn,inputs,output):
  inn = dict()
  a = dict()
  # Copy inputs to 0..n
  for i in range(0,len(inputs)):
    a[i] = inputs[i]
  a[-1] = 1
  # Propagate the input values forward
  for level in range(1,nn.lastLevel+1):
    for j in nn.nodesOfLevel(level):
      inn[j] = sum([a[i]*nn.getWeight(i,j) for i in nn.incoming(j)])
      a[j] = g(inn[j])
  lvl = nn.nodesOfLevel(nn.lastLevel);
  return a[lvl[output]]
