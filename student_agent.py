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
    """
    This function MUST return an integer action in [0..5].

    1) Convert obs to a key
    2) If key is in Q_TABLE, pick the best action
    3) Otherwise, fallback to random
    """
    
    state_key = make_discrete_key(obs)

    if state_key in Q_TABLE:
        q_values = Q_TABLE[state_key]
        return int(np.argmax(q_values))
    else:
        # Fallback (unseen state) -> random
        return random.choice([0, 1, 2, 3, 4, 5])
