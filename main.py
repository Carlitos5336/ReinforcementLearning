import numpy as np
import random as rd

INF = 1e9

#---------------------|
#       CLASSES       |
#---------------------|

class Action:
    
    def __init__(self, name):
        
        self.name = name

    def __repr__(self):
        
        return self.name

class Transition:
    
    def __init__(self, action, state, probability, reward):
        
        self.action = action
        self.state = state
        self.probability = probability
        self.reward = reward
        
    def getState(self):
        
        return self.state
    
    def getProbability(self):
        
        return self.probability
    
    def getReward(self):
        
        return self.reward
    
    def getAction(self):
        
        return self.action
    
    def __repr__(self):
        
        return str([self.state, self.probability, self.reward])

class State:
    
    counter = 0
    
    def __init__(self, name):
        
        self.name = name
        self.transitions = {}
        self.number = State.counter
        State.counter += 1
        
    def addTransition(self, transition):
        
        """
        Uses action as key access to the transitions of that state-action
        """
        
        action = transition.getAction()
        
        if action in self.transitions.keys():
            self.transitions[action].append(transition)
        else:
            self.transitions[action] = [transition]
            
    def getTransitions(self):
        
        return self.transitions
            
    def getPossibleActions(self):
        
        return list(set(self.transitions.keys()))
    
    def getNumber(self):
        
        return self.number
    
    def useAction(self, action):
        
        """
        Receives an action and return the transition that this action leads to from this state according to its probability
        """
        
        possibleTransitions = self.transitions[action]
        probabilities = [transition.getProbability() for transition in possibleTransitions]
        next_transition = np.random.choice(possibleTransitions, p = probabilities)
        
        return next_transition
    
    def __repr__(self):
        
        return self.name
    
class Policy:
    
    def __init__(self, values, states, actions):
        """
        Receives values, a dict mapping each state with an action to take.
                 states, a list of all system states
                 actions, a list of all system actions
        """
        
        self.states = states
        self.actions = actions
        self.values = values
        
    def clone(self):
        
        return Policy(self.values.copy(), self.states, self.actions)
        
    def getAllActions(self):
        
        return self.actions
    
    def getValues(self):
        
        return self.values
        
    def RPI(self):
        """
        Returns inmediate reward matrix pi

        """
        
        rpi = np.zeros((len(self.states), 1))
        for state, action in self.values.items():
            rpi[state.getNumber()] = getExpectedReward(action, state)
            
        return rpi

    def PPI(self):
        """
        Returns probability matrix pi

        """
        
        ppi = np.zeros((len(self.states), len(self.states)))
        for state, action in self.values.items():
            ppi[state.getNumber()] = getProbabilityMatrix(action, state, len(self.states))
            
        return ppi
    
    def VPI(self, lambda_):
        """
        Returns value function matrix pi

        """
        
        rpi= self.RPI()
        ppi = self.PPI()
        I = np.identity(len(self.states))
        vpi = np.matmul(np.linalg.inv(I - ppi * lambda_), rpi)
        
        return vpi

    def __eq__(self, other):
        
        if other == None:
            return False
        
        return self.values == other.values
    
#----------------------------|
#      OTHER FUNCTIONS       |
#----------------------------|

def getExpectedReward(action, state):
    
    if action not in state.getTransitions().keys():
        return -INF
    
    expected_reward = 0
    for transition in state.getTransitions()[action]:
        expected_reward += transition.getReward() * transition.getProbability()
    
    return expected_reward

def getProbabilityMatrix(action, state, numberOfStates):
    
    if action not in state.getTransitions().keys():
        return np.zeros((1, numberOfStates))
    
    probabilities = np.zeros((1, numberOfStates))
    for transition in state.getTransitions()[action]:
        probabilities[0, transition.getState().getNumber()] = transition.getProbability()
        
    return probabilities
    
#----------------------------|
#       MAIN FUNCTIONS       |
#----------------------------|      
    
def qLearning(initial_state, actions, lambda_, delta = 1e-8, epsilon = 0):
    
    state = initial_state
    actions = [Slow, Fast]
    
    curr_qvalues = {state : {}}
    
    for action in actions:
        curr_qvalues[state][action] = 0
        
    counter = 0
        
    def convergence():
        
        return counter == 1000
    
    while not convergence():
        
        counter += 1
        print("QLearning iteration", counter, "...")
        
        max_qvalue = -INF
        best_action = None
        
        for action in state.getPossibleActions():
            qvalue = curr_qvalues[state][action]
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                best_action = action
                
        if rd.random() < epsilon:
            #print("Taking a random action...")
            best_action = rd.choice(state.getPossibleActions())
        
        transition = state.useAction(best_action)
        inmediateReward = transition.getReward()
        next_state = transition.getState()
        
        #print("")
        #print("QValues:", curr_qvalues)
        #print("Current state:", state)
        #print("Action taken:", best_action)
        #print("Next state:", next_state)
        #input()
    
        if next_state not in curr_qvalues.keys():
            curr_qvalues[next_state] = {}
            for action in actions:
                curr_qvalues[next_state][action] = 0
           
        max_qvalue = -INF
        best_next_action = None
        
        for action in next_state.getPossibleActions():
            qvalue = curr_qvalues[next_state][action]
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                best_next_action = action
        
        curr_qvalues[state][best_action] = inmediateReward + lambda_ * curr_qvalues[next_state][best_next_action]
        
        state = next_state
        
    return curr_qvalues

def valueIteration(states, actions, lambda_, delta = 1e-8):
    
    curr_value = np.random.rand(len(states), 1)
    last_value = None
    counter = 0
    
    def convergence():
        
        if last_value is None:
            return False
        
        diff = curr_value - last_value
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if diff[i, j] >= delta:
                    return False
            
        return True
    
    while not convergence():
    
        counter += 1
        print("Value iteration", counter, "...")
        
        last_value = curr_value.copy()
    
        for state in states:
            
            best_reward = -INF
            
            for action in actions:
                
                inmediateReward = getExpectedReward(action, state)
                probabilityMatrix = getProbabilityMatrix(action, state, len(states))
                totalReward = (inmediateReward + lambda_ * np.matmul(probabilityMatrix, last_value))[0]
                
                if totalReward > best_reward:
                    best_reward = totalReward
            
            curr_value[state.getNumber()] = best_reward
    
    return last_value

def policyIteration(initial_policy, lambda_):
    
    curr_policy = initial_policy
    last_policy = None
    counter = 0
    
    while last_policy != curr_policy:
    
        counter += 1
        print("Policy iteration", counter, "...")
        
        last_policy = curr_policy.clone()
        valueFunction = curr_policy.VPI(lambda_)
        
        for state in curr_policy.values.keys():
            
            best_action = None
            best_reward = -INF
            
            for action in curr_policy.getAllActions():
                
                inmediateReward = getExpectedReward(action, state)
                probabilityMatrix = getProbabilityMatrix(action, state, len(states))
                totalReward = (inmediateReward + lambda_ * np.matmul(probabilityMatrix, valueFunction))[0]
                
                if totalReward > best_reward:
                    best_reward = totalReward
                    best_action = action
            
            curr_policy.getValues()[state] = best_action
    
    return last_policy

#-------------------|
#       MAIN        |
#-------------------|

# Create actions
Slow = Action("Slow")
Fast = Action("Fast")
actions = [Slow, Fast]

# Create states
Fallen = State("Fallen")
Standing = State("Standing")
Moving = State("Moving")
states = [Fallen, Standing, Moving]

# Create transitions
Fallen.addTransition(Transition(Slow, Standing, 0.4, +1))
Fallen.addTransition(Transition(Slow, Fallen, 0.6, -1))
Standing.addTransition(Transition(Slow, Moving, 1, +1))
Standing.addTransition(Transition(Fast, Fallen, 0.4, -1))
Standing.addTransition(Transition(Fast, Moving, 0.6, +2))
Moving.addTransition(Transition(Slow, Moving, 1, +1))
Moving.addTransition(Transition(Fast, Fallen, 0.2, -1))
Moving.addTransition(Transition(Fast, Moving, 0.8, +2))

lambda_ = 0.1

policy = Policy({Fallen : Slow, Standing : Slow, Moving : Slow}, states, actions)

print(policyIteration(policy, lambda_).VPI(lambda_))
print("")
print(valueIteration(states, actions, lambda_))
print("")
print(qLearning(Fallen, actions, lambda_, epsilon = 0.1))