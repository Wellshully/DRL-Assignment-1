# Filename: student_agent.py

import pickle
import os
import numpy as np
import random



Q_TABLE = {}
q_table_filename = "q_table_rule.pkl"

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
import random

# Global variable to track if the taxi is carrying the passenger.
# (This should be maintained by your agent across steps.)
CARRYING = False

def get_action(obs):
    """
    Rule-based action selection using the observation:
    
    The observation (state) is assumed to be a tuple with the following format:
      (taxi_row, taxi_col,
       station0_x, station0_y, station1_x, station1_y,
       station2_x, station2_y, station3_x, station3_y,
       obstacle_north, obstacle_south, obstacle_east, obstacle_west,
       passenger_look, destination_look)
       
    Action mapping:
      0: Move South
      1: Move North
      2: Move East
      3: Move West
      4: PICKUP
      5: DROPOFF
    """
    global CARRYING

    # Unpack observation values
    taxi_row, taxi_col = obs[0], obs[1]
    # Fixed station coordinates:
    stations = [(obs[2], obs[3]), (obs[4], obs[5]),
                (obs[6], obs[7]), (obs[8], obs[9])]
    # Obstacle flags:
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east  = obs[12]
    obstacle_west  = obs[13]
    # Look flags:
    passenger_look   = obs[14]
    destination_look = obs[15]
    
    # --- Helper: Choose a movement action that moves closer to a target coordinate ---
    def move_toward(target):
        target_row, target_col = target
        # Compute differences
        d_row = target_row - taxi_row
        d_col = target_col - taxi_col
        
        # Decide on prioritized movement: choose the axis with the larger difference.
        if abs(d_row) >= abs(d_col):
            # Try vertical move first.
            if d_row > 0 and obstacle_south == 0:
                return 0  # Move South
            elif d_row < 0 and obstacle_north == 0:
                return 1  # Move North
            # If vertical is blocked or zero, try horizontal.
            if d_col > 0 and obstacle_east == 0:
                return 2  # Move East
            elif d_col < 0 and obstacle_west == 0:
                return 3  # Move West
        else:
            # Try horizontal move first.
            if d_col > 0 and obstacle_east == 0:
                return 2  # Move East
            elif d_col < 0 and obstacle_west == 0:
                return 3  # Move West
            # Then vertical.
            if d_row > 0 and obstacle_south == 0:
                return 0  # Move South
            elif d_row < 0 and obstacle_north == 0:
                return 1  # Move North
        
        # If the prioritized direction is blocked, return any safe move.
        safe_moves = []
        if obstacle_south == 0:
            safe_moves.append(0)
        if obstacle_north == 0:
            safe_moves.append(1)
        if obstacle_east == 0:
            safe_moves.append(2)
        if obstacle_west == 0:
            safe_moves.append(3)
        if safe_moves:
            return random.choice(safe_moves)
        return None  # no safe move found
    
    # --- Rule-based decisions ---
    
    # If not carrying passenger, try to pick up if near passenger.
    if not CARRYING:
        # Look for a station that is adjacent or on the same cell and could be the passenger location.
        for station in stations:
            s_row, s_col = station
            # Check if the taxi is exactly at the station.
            if (taxi_row, taxi_col) == (s_row, s_col):
                if passenger_look:
                    # We are at the passenger location: pick up.
                    CARRYING = True  # Update our flag.
                    return 4  # PICKUP
            else:
                # If adjacent (Manhattan distance 1) and passenger_look is true, head toward that station.
                if abs(taxi_row - s_row) + abs(taxi_col - s_col) == 1 and passenger_look:
                    move = move_toward(station)
                    if move is not None:
                        return move
        # If passenger_look is false or we are not near any station,
        # use a fallback strategy: e.g., move randomly (but only among safe moves).
        safe_moves = []
        if obstacle_south == 0:
            safe_moves.append(0)
        if obstacle_north == 0:
            safe_moves.append(1)
        if obstacle_east == 0:
            safe_moves.append(2)
        if obstacle_west == 0:
            safe_moves.append(3)
        if safe_moves:
            return random.choice(safe_moves)
        else:
            return random.choice([0,1,2,3])
    
    # If carrying the passenger, aim for destination.
    else:
        # Look for a station that is adjacent or on the same cell and could be the destination.
        for station in stations:
            s_row, s_col = station
            # Check if the taxi is exactly at the destination station.
            if (taxi_row, taxi_col) == (s_row, s_col):
                if destination_look:
                    # We are at the destination: drop off.
                    CARRYING = False  # Reset our flag.
                    return 5  # DROPOFF
            else:
                # If adjacent (Manhattan distance 1) and destination_look is true, head toward that station.
                if abs(taxi_row - s_row) + abs(taxi_col - s_col) == 1 and destination_look:
                    move = move_toward(station)
                    if move is not None:
                        return move
        # Fallback when not near destination: move randomly among safe directions.
        safe_moves = []
        if obstacle_south == 0:
            safe_moves.append(0)
        if obstacle_north == 0:
            safe_moves.append(1)
        if obstacle_east == 0:
            safe_moves.append(2)
        if obstacle_west == 0:
            safe_moves.append(3)
        if safe_moves:
            return random.choice(safe_moves)
        else:
            return random.choice([0,1,2,3])

