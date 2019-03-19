
import math
import random

from backprogexamples import *


learningrate = 1

def backpropagation(nn,inputs,outputs):
  inn = dict()
  a = dict()
  Delta = dict()
### INSERT YOUR CODE HERE
### Use the functions in backprogexamples.py to access and modify
### the neural network. The grader will read the training result
### from that representation.
  def g(inn):   
    return 1/(1+math.exp(-inn));
  def diff_g(inn):
    return g(inn)*(1-g(inn))
  
  #forward propagation 
  for i in range(0,len(inputs)):
    a[i] = inputs[i]
  #Bias term  
  a[-1] = 1
  # Propagate the input values forward
  for level in range(1,nn.lastLevel+1):
    for j in nn.nodesOfLevel(level):
      inn[j] = sum([a[i]*nn.getWeight(i,j) for i in nn.incoming(j)])
      a[j] = g(inn[j])
  lvl = nn.nodesOfLevel(nn.lastLevel)
  calculated_output = [ a[lvl[o]] for o in range(len(nn.outputs))]

  
  for index,j in enumerate(nn.outputs):
    Delta[j] = diff_g(inn[j])*(outputs[index]- calculated_output[index])
  
  for level in range(nn.lastLevel-1,0,-1):
    for i in nn.nodesOfLevel(level):
      # print(f"level: {level} and {i}")
      Delta[i] = diff_g(inn[i]) * sum( nn.w[i,j]*Delta[j] for j in nn.successors[i])
    
  #Update weights 
  for i,j in nn.w.keys() :
    nn.w[i,j] =  nn.w[i,j] + learningrate*a[i]*Delta[j]
  
