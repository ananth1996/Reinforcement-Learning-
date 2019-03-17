
import random

from qlearnexamples import *

# The Q-Learning Algorithm

# EXERCISE ASSIGNMENT:
# Implement the Q-learning algorithm for MDPs.
#   The Q-values are represented as a Python dictionary Q[s,a],
# which is a mapping from the state indices s=0..stateMax to
# and actions a to the Q-values.
#
# Choice of actions can be completely random, or, if you are interested,
# you could implement some scheme that prefers better actions, e.g.
# based on Multi-arm Bandit problems (find more about these in the literature:
# this is an optional addition to the programming assignment.)

# OPTIONAL FUNCTIONS:
# You can implement and use the auxiliary functions bestActionFor and execute
# if you want, as auxiliary functions for Qlearning and makePolicy and makeValues.

# bestActionFor chooses the best action for 'state', given Q values

def bestActionFor(mdp,state,Q):
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE

# valueOfBestAction gives the value of best action for 'state'

def valueOfBestAction(mdp,state,Q):
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE

# 'execute' randomly chooses a successor state for state s w.r.t. action a.
# The probability with which is given successor is chosen must respect
# to the probability given by mdp.successors(s,a).
# It returns a tuple (s2,r), where s2 is the successor state and r is
# the reward that was obtained.

def execute(mdp,s,a):
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE


# OBLIGATORY FUNCTION:
# Qlearning returns the Q-value function after performing the given
#   number of iterations i.e. Q-value updates.

def Qlearning(mdp,gamma,lambd,iterations):
  # The Q-values are a real-valued dictionary Q[s,a] where s is a state and a is an action.
  Q = dict()
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  return Q

# OBLIGATORY FUNCTION:
# makePolicy constructs a policy, i.e. a mapping from state to actions,
#   given a Q-value function as produced by Qlearning.

def makePolicy(mdp,Q):
  # A policy is an action-valued dictionary P[s] where s is a state
  P = dict()
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  return P


# OBLIGATORY FUNCTION:
# makeValues constructs the value function, i.e. a mapping from states to values,
#   given a Q-value function as produced by Qlearning.

def makeValues(mdp,Q):
  # A value function is a real-valued dictionary V[s] where s is a state
  V = dict()
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  return V
