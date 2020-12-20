# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        dirct = util.Counter()
        for i in range(self.iterations):
            for state in self.mdp.getStates():
                # get best action
                action = self.computeActionFromValues(state)
                # if there is no action to do, set value to zero.
                value = 0
                if action:
                    value = self.computeQValueFromValues(state, action)
                dirct[state] = value
            self.values = dirct.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qvalue = 0
        # get the list of nextstate and its probability.
        nextinformation = self.mdp.getTransitionStatesAndProbs(state, action)
        # compute qvalue, do add.
        for nextstate, prob in nextinformation:
            temp = self.mdp.getReward(state, action, nextstate)
            qvalue += prob * (temp + self.discount * self.getValue(nextstate))
        return qvalue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # get the actions you can do in current state.
        actions = self.mdp.getPossibleActions(state)
        # if there is no action to do, we return none.
        if len(actions) == 0:
            return None
        maxvalue = float('-Inf')
        bestacton = actions[0]
        #the higher the value, the better the action.
        for action in actions:
            tempvalue = self.computeQValueFromValues(state, action)
            if tempvalue > maxvalue:
                maxvalue = tempvalue
                bestacton = action
        return bestacton
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
