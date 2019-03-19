
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
  best_action = None
  best_Q = -float("inf")
  for action in mdp.applicableActions(state):
    if Q[state,action] > best_Q:
      best_Q = Q[state,action]
      best_action = action
  
  return best_action

# valueOfBestAction gives the value of best action for 'state'

def valueOfBestAction(mdp,state,Q):
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  best_Q = -float("inf")
  for action in mdp.applicableActions(state):
    best_Q = max(best_Q,Q[state,action])
    # print(Q[state,action])
  return best_Q

# 'execute' randomly chooses a successor state for state s w.r.t. action a.
# The probability with which is given successor is chosen must respect
# to the probability given by mdp.successors(s,a).
# It returns a tuple (s2,r), where s2 is the successor state and r is
# the reward that was obtained.

def execute(mdp,s,a):
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  tuples = mdp.successors(s,a)
  successors, probs, reward = zip(*tuples)
  # Group the state and reward together again
  state_reward = list(zip(*[successors,reward]))
  # Choose a state and reward according to the probabilities
  rand = random.random()
  for i,p in enumerate(probs):
    if rand < p:
      return state_reward[i]
    rand -= p
  # choice = random.choices(population=state_reward,weights=probs,k=1)[0]
  # return choice

# OBLIGATORY FUNCTION:
# Qlearning returns the Q-value function after performing the given
#   number of iterations i.e. Q-value updates.

def Qlearning(mdp,gamma,lambd,iterations):
  # The Q-values are a real-valued dictionary Q[s,a] where s is a state and a is an action.
  Q = dict()
### YOUR CODE HERE
### YOUR CODE HERE
### YOUR CODE HERE
  num_states = mdp.stateMax
  actions = mdp.ACTIONS
  for s in range(0,num_states+1):
    for a in actions:
      Q[s,a]=0
  
  #starting state
  s = random.choice(range(0,num_states+1))
  for _ in range(iterations):
    a = random.choice(mdp.applicableActions(s))
    next_s,r = execute(mdp,s,a)
    Q[s,a] = (1-lambd)*Q[s,a] + lambd*(r+ gamma* valueOfBestAction(mdp,next_s,Q))
    s = next_s
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
  for s in range(0,mdp.stateMax+1):
    P[s] = bestActionFor(mdp,s,Q)
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
  for s in range(0,mdp.stateMax+1):
    # print(f"state{s}")
    val = valueOfBestAction(mdp,s,Q)
    val = max(val,0)
    V[s]=val
  return V
