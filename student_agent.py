# Filename: student_agent.py

import pickle
import os
import numpy as np
import random

# Try to load Q-table from pickle
Q_TABLE = {}
q_table_filename = "q_table.pkl"

if os.path.exists(q_table_filename):
    with open(q_table_filename, "rb") as f:
        Q_TABLE = pickle.load(f)
else:
    # If Q-table file isn't found, we can keep Q_TABLE empty 
    # and fallback to random actions (or any default).
    Q_TABLE = {}

def make_discrete_key(obs):
    """
    Convert environment's observation to a discrete dict key.
    Must match the encoding used during training!
    """
    return tuple(obs)

def get_action(obs):
    obstacle_north = obs[10]  # for action 1 (Move North)
    obstacle_south = obs[11]  # for action 0 (Move South)
    obstacle_east  = obs[12]  # for action 2 (Move East)
    obstacle_west  = obs[13]  # for action 3 (Move West)
    """
    This function MUST return an integer action in [0..5].

    1) Convert obs to a key
    2) If key is in Q_TABLE, pick the best action
    3) Otherwise, fallback to random
    """
    
    state_key = make_discrete_key(obs)
    def is_blocked(action, obs):
        if action == 0:  # Move South
            return obs[11] == 1
        elif action == 1:  # Move North
            return obs[10] == 1
        elif action == 2:  # Move East
            return obs[12] == 1
        elif action == 3:  # Move West
            return obs[13] == 1
        return False  # Actions 4 and 5 are not movement actions.
    if state_key in Q_TABLE:
        q_values = Q_TABLE[state_key]
        act= int(np.argmax(q_values))
        if act in [0, 1, 2, 3] and is_blocked(act, obs):
            safe_actions = []
            for action in [0, 1, 2, 3]:
                if not is_blocked(action, obs):
                    safe_actions.append(action)
           
            safe_actions.extend([4, 5])
            
            act = max(safe_actions, key=lambda a: q_values[a])
        
        return act
    else:
        safe_actions = []
        for action in [0, 1, 2, 3]:
            if not is_blocked(action, obs):
                safe_actions.append(action)
        safe_actions.extend([4, 5])
        return random.choice(safe_actions)
