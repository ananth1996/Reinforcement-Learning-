
import math
import random

import matplotlib.pyplot as plt

from backprogexamples import *
from backprog import *

# Training the neural network
# There is two *very* simple functions to learn:
# 1 bump: Output 1 in the interval 0.5..1, and output 0 elsewhere.
# 2 bumps: Output 1 in the intervals 0.5..1 and 1.5..2, and output 0 elsewhere.

fig = plt.figure(1)

def targetFn(x, bumps):
  if bumps == 1 and x > 0.5 and x < 1:
    y = 1
  elif bumps == 2 and ((x > 0.5 and x < 1) or  (x > 1.5 and x < 2)):
    y = 1
  else:
    y = 0

  return y

def train(nn,bumps):
  # Do lots of training
  for i in range(0,100000):
    # pick an input output pair, randomly (output is 1 for the interval ]2,3[)
    x = random.uniform(-1,3)
    y = targetFn(x, bumps)
    backpropagation(nn,[x],[y])

def frange(start, stop, step):
  i = start
  while i < stop:
    yield i
    i += step

# Create a neural network
# layer 0: 0 (one input)
# layer 1: 1 2 3 4
# layer 2: 5 (one output)
nn1 = NeuralNetwork(1,[[1,2,3,4],[5]],-1,1);

# Simpler network, with only 3 units
nn0 = NeuralNetwork(1,[[1,2],[3]],-1,1);

# The neural network with 3 nodes can fairly well represent the 1-bump
# function, and the 5-node network can represent the 2-bump function.
# In the 2-bump 5-node case, backpropagation does not always manage to
# converge to the desired function, dependent on how the initial randomization
# of connection weights was. I am not sure what the explanation exactly is.

def showOutputs(nn, bumps, label):
  in0 = [x for x in frange(0,2.5,0.05)]
  out = [evaluate(nn,[x],0) for x in in0]
  plt.plot(in0, out, label=label)

def plotCorrectValues(nn, bumps):
  in0 = [x for x in frange(0,2.5,0.05)]
  out = [targetFn(x,bumps) for x in in0]
  plt.plot(in0, out, label='correct')

def showWeights(nn):
  for j in range(nn.firstNeuron,nn.lastNeuron+1):
    for i in nn.incoming(j):
      print("Weight " + str(i) + "," + str(j) + " = " + str(nn.getWeight(i,j)))

def tryout(nn,bumps):
  plotCorrectValues(nn, bumps)
  showOutputs(nn, bumps, 'before')
  train(nn,bumps)
  showOutputs(nn, bumps, 'after')
  # showWeights(nn)


# tryout(nn0,1)
tryout(nn1,2)

plt.legend()
plt.show()
