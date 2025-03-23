import pickle
import os
import numpy as np
import random
from collections import defaultdict
Q_TABLE = {}
q_table_filename = "q_table.pkl"

if os.path.exists(q_table_filename):
    with open(q_table_filename, "rb") as f:
        Q_TABLE = pickle.load(f)
else:
    Q_TABLE = {}

CARRYING = False
passenger_station = None
destination_station = None
visited_stations = set()
tracking_path = False
path_to_passenger = []
def make_discrete_key(obs):
    return tuple(obs)

action_memory = {
    'find_passenger': defaultdict(set),
    'go_to_destination': defaultdict(set)
}

def move_toward(taxi_row, taxi_col, target, obstacle_flags, closer_move_prob=0.8, used_action_penalty=0.01):
    global CARRYING,  action_memory
    carrying = CARRYING
    state_key = 'go_to_destination' if carrying else 'find_passenger'
   
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obstacle_flags
    target_row, target_col = target

    current_dist = abs(target_row - taxi_row) + abs(target_col - taxi_col)

    action_to_pos = {
        0: (taxi_row + 1, taxi_col),  # South
        1: (taxi_row - 1, taxi_col),  # North
        2: (taxi_row, taxi_col + 1),  # East
        3: (taxi_row, taxi_col - 1),  # West
    }

   
    safe_moves = []
    for action, (new_row, new_col) in action_to_pos.items():
        if (action == 0 and obstacle_south) or \
           (action == 1 and obstacle_north) or \
           (action == 2 and obstacle_east) or \
           (action == 3 and obstacle_west):
            continue  
        safe_moves.append(action)

    if not safe_moves:
        return random.randint(0, 3) 

    
    closer_moves = []
    non_closer_moves = []
    for action in safe_moves:
        new_row, new_col = action_to_pos[action]
        new_dist = abs(target_row - new_row) + abs(target_col - new_col)
        if new_dist < current_dist:
            closer_moves.append(action)
        else:
            non_closer_moves.append(action)
    
    prob_dist = []
    moves_considered = safe_moves  # all safe moves
    for action in moves_considered:
        if action in closer_moves:
            prob_dist.append(closer_move_prob)
        else:
            prob_dist.append(1 - closer_move_prob)

   
    position_key = (taxi_row, taxi_col)
    actions_used_here = action_memory[state_key][position_key]

    for idx, action in enumerate(moves_considered):
        if action in actions_used_here:
            prob_dist[idx] *= used_action_penalty  
            

    # Normalize probabilities
    total_prob = sum(prob_dist)
    if total_prob == 0:
        prob_dist = [1 / len(prob_dist)] * len(prob_dist)  # uniform if all penalties zero out
    else:
        prob_dist = [p / total_prob for p in prob_dist]

   
    chosen_action = random.choices(moves_considered, weights=prob_dist, k=1)[0]

   
    action_memory[state_key][position_key].add(chosen_action)

    return chosen_action

def get_action(obs):
    global CARRYING, passenger_station, destination_station, visited_stations
    global tracking_path, path_to_passenger
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    obstacle_flags = (obs[10], obs[11], obs[12], obs[13])
    passenger_look = obs[14]
    destination_look = obs[15]

    current_pos = (taxi_row, taxi_col)

  
    if current_pos in stations:
        if not passenger_look and not destination_look:
            visited_stations.add(current_pos)

   
    if passenger_look:
        for station in stations:
            if abs(taxi_row - station[0]) + abs(taxi_col - station[1]) <= 1:
                passenger_station = station
                visited_stations.add(station)
                break

    
    if destination_look:
        for station in stations:
            if abs(taxi_row - station[0]) + abs(taxi_col - station[1]) <= 1:
                destination_station = station
                visited_stations.add(station)
                if not CARRYING and not tracking_path:
                    tracking_path = True
                    path_to_passenger = [] 
                break
    if current_pos in path_to_passenger:
        first_idx = path_to_passenger.index(current_pos)
        
        path_to_passenger = path_to_passenger[:first_idx+1]
    else:
        path_to_passenger.append(current_pos)
    
    if not CARRYING:
        if passenger_station:
            if current_pos == passenger_station:
                CARRYING = True
                return 4  # PICKUP
            else:
                return move_toward(taxi_row, taxi_col, passenger_station, obstacle_flags)
        else:
            
            unvisited_stations = [s for s in stations if s not in visited_stations]
            if unvisited_stations:
                nearest_station = min(unvisited_stations, key=lambda s: abs(taxi_row - s[0]) + abs(taxi_col - s[1]))
                return move_toward(taxi_row, taxi_col, nearest_station, obstacle_flags)
            else:
                
                safe_moves = [a for a, blocked in zip([0,1,2,3], obstacle_flags) if not blocked]
                return random.choice(safe_moves) if safe_moves else random.randint(0, 3)
    else:
        if destination_station:
            if current_pos == destination_station:
                CARRYING = False
                passenger_station = None
                destination_station = None
                tracking_path = False
                path_to_passenger.clear()
                visited_stations.clear() 
                return 5  # DROPOFF
            elif path_to_passenger:
                idx = path_to_passenger.index(current_pos)
                if idx > 0:
                    next_pos = path_to_passenger[idx - 1]
                    dr = next_pos[0] - taxi_row
                    dc = next_pos[1] - taxi_col
                    if dr == -1:
                        return 1  # Move North
                    elif dr == 1:
                        return 0  # Move South
                    elif dc == 1:
                        return 2  # Move East
                    elif dc == -1:
                        return 3  # Move West
            return move_toward(taxi_row, taxi_col, destination_station, obstacle_flags)
        else:
            unvisited_stations = [s for s in stations if s not in visited_stations]
            if unvisited_stations:
                nearest_station = min(unvisited_stations, key=lambda s: abs(taxi_row - s[0]) + abs(taxi_col - s[1]))
                return move_toward(taxi_row, taxi_col, nearest_station, obstacle_flags)
            else:
                safe_moves = [a for a, blocked in zip([0,1,2,3], obstacle_flags) if not blocked]
                return random.choice(safe_moves) if safe_moves else random.randint(0, 3)