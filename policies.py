import numpy as np
import random
from collections import deque


        
class UnifRandomPolicy:
    def __init__(self, env):
        self.env = env
        self.goal = None
        self.actions = list(env.actions.keys())

    def act(self, state):    
        # Filter valid actions in current state
        valid_actions = [a for a in self.actions if self.env.is_valid_transition(state, a)[0]]
        
        if not valid_actions:
            # No valid actions (shouldn't happen in well-formed env)
            return None, False
        
        # Uniform random over valid actions
        return random.choice(valid_actions), True
    
    def pi_probability(self, state, action):
        valid_actions = [a for a in self.actions if self.env.is_valid_transition(state, a)[0]]
        return 1.0/len(valid_actions)
    
   
    

        
class GoalDirectedPolicy:
    def __init__(self, env, goal_state, epsilon=0.2):
        self.env = env
        self.epsilon = epsilon
        self.goal = goal_state
        self.actions = list(env.actions.keys())
        self.path = []


    def _bfs(self, start):
        visited = set()
        queue = deque([(start, [])])
        while queue:
            state, path = queue.popleft()
            if state == self.goal:
                return path  # list of actions
            if state in visited:
                continue
            visited.add(state)
            
            for a in self.actions:
                valid_step, neighbor = self.env.is_valid_transition(state, a)
                if valid_step: queue.append((neighbor, path + [a])) 
            
        return []  # no path found


    def act(self, state):
        if state == self.goal:
            return 0, True
        
        self.path = self._bfs(state)
        
        # With probability epsilon, act randomly
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Otherwise, follow path
        next_action = self.path.pop(0)
        return next_action, False
    
    
    
    
    
class AveragedGoalPolicy:
    def __init__(self, env, num_goals, epsilon=0.1):
        self.env = env
        self.goal = None
        self.epsilon = epsilon
        self.actions = list(env.actions.keys())
        self.num_states = len(env.state_to_idx)
        self.num_actions = len(self.actions)
        self.num_goals = num_goals
        self.policy_matrix = np.zeros((self.num_states, self.num_actions))  # π(a|s)
        self._precompute_policy()

    def _precompute_policy(self):
        if self.num_goals is not None:
            goals = self.env.random_valid_position(self.num_goals)
        else:
            goals = [self.env.idx_to_state[i] for i in range(self.num_states)]

        for state_idx in range(self.num_states):
            state = self.env.idx_to_state[state_idx]
            valid_actions = [a for a in self.actions if self.env.is_valid_transition(state, a)[0]]
            action_counts = {a: 0 for a in valid_actions}

            for goal in goals:
                path = GoalDirectedPolicy(self.env, goal, epsilon=0)._bfs(state)
                if path:
                    optimal_action = path[0]
                    if optimal_action in action_counts:
                        action_counts[optimal_action] += 1

            total = sum(action_counts.values())
            for a in valid_actions:
                avg_prob = action_counts[a] / self.num_states if total > 0 else 0.0
                # Apply epsilon smoothing
                self.policy_matrix[state_idx, a] = (1 - self.epsilon) * avg_prob + self.epsilon / len(valid_actions)
            self.policy_matrix[state_idx,:] /= sum(self.policy_matrix[state_idx,:])

    def act(self, state):
        return np.random.choice(self.actions, p=self.policy_matrix[self.env.state_to_idx[state], :]), True
    
    def pi_probability(self, state, action):
        return self.policy_matrix[self.env.state_to_idx[state], action]
    
    
    
    
class GreedySMPolicy:
    def __init__(self, env, M, goal_state):
        self.env = env
        self.M = M  # occupancy matrix
        self.actions = list(env.actions.keys())
        self.goal = goal_state


    def act(self, state):
        # For each action, get M((s,a),(sg,ag))
        scores = []
        for a in self.actions:
            idx = self.env.sa_index(state, a)
            score = sum(self.M[idx, [self.env.sa_index(self.goal, a) for a in self.actions]])
            scores.append((a, score))
        # Choose action with highest score
        if max(scores) == 0:
            return None, False
        best_action = max(scores, key=lambda x: x[1])[0]
        return best_action, True