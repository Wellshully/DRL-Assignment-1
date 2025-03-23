import pickle
import os
import numpy as np
import random

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

def make_discrete_key(obs):
    return tuple(obs)



def move_toward(taxi_row, taxi_col, target, obstacle_flags, closer_move_prob=0.6):
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obstacle_flags
    target_row, target_col = target

    current_dist = abs(target_row - taxi_row) + abs(target_col - taxi_col)

    action_to_pos = {
        0: (taxi_row + 1, taxi_col),  # South
        1: (taxi_row - 1, taxi_col),  # North
        2: (taxi_row, taxi_col + 1),  # East
        3: (taxi_row, taxi_col - 1),  # West
    }

    # Step 1: Find all safe moves
    safe_moves = []
    for action, (new_row, new_col) in action_to_pos.items():
        if (action == 0 and obstacle_south) or \
           (action == 1 and obstacle_north) or \
           (action == 2 and obstacle_east) or \
           (action == 3 and obstacle_west):
            continue  # Skip blocked moves
        safe_moves.append(action)

    if not safe_moves:
        return random.randint(0, 3)  # no safe moves, choose random

    # Step 2: Classify moves into closer and non-closer
    closer_moves = []
    non_closer_moves = []
    for action in safe_moves:
        new_row, new_col = action_to_pos[action]
        new_dist = abs(target_row - new_row) + abs(target_col - new_col)
        if new_dist < current_dist:
            closer_moves.append(action)
        else:
            non_closer_moves.append(action)

    # Step 3: Choose probabilistically
    if closer_moves and non_closer_moves:
        # Assign probabilities
        prob_dist = []
        closer_weight = closer_move_prob / len(closer_moves)
        non_closer_weight = (1 - closer_move_prob) / len(non_closer_moves)

        for action in safe_moves:
            if action in closer_moves:
                prob_dist.append(closer_weight)
            else:
                prob_dist.append(non_closer_weight)

        chosen_action = random.choices(safe_moves, weights=prob_dist, k=1)[0]

    elif closer_moves:
        chosen_action = random.choice(closer_moves)
    else:
        chosen_action = random.choice(non_closer_moves)

    return chosen_action

def get_action(obs):
    global CARRYING, passenger_station, destination_station, visited_stations

    taxi_row, taxi_col = obs[0], obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    obstacle_flags = (obs[10], obs[11], obs[12], obs[13])
    passenger_look = obs[14]
    destination_look = obs[15]

    current_pos = (taxi_row, taxi_col)

    # Check if at station and mark as visited if neither passenger nor destination
    if current_pos in stations:
        if not passenger_look and not destination_look:
            visited_stations.add(current_pos)

    # Update passenger station if detected
    if passenger_look:
        for station in stations:
            if abs(taxi_row - station[0]) + abs(taxi_col - station[1]) <= 1:
                passenger_station = station
                visited_stations.add(station)
                break

    # Update destination station if detected
    if destination_look:
        for station in stations:
            if abs(taxi_row - station[0]) + abs(taxi_col - station[1]) <= 1:
                destination_station = station
                visited_stations.add(station)
                break

    # Action logic
    if not CARRYING:
        if passenger_station:
            if current_pos == passenger_station:
                CARRYING = True
                return 4  # PICKUP
            else:
                return move_toward(taxi_row, taxi_col, passenger_station, obstacle_flags)
        else:
            # Go to nearest unvisited station
            unvisited_stations = [s for s in stations if s not in visited_stations]
            if unvisited_stations:
                nearest_station = min(unvisited_stations, key=lambda s: abs(taxi_row - s[0]) + abs(taxi_col - s[1]))
                return move_toward(taxi_row, taxi_col, nearest_station, obstacle_flags)
            else:
                # All stations visited, move randomly safely
                safe_moves = [a for a, blocked in zip([0,1,2,3], obstacle_flags) if not blocked]
                return random.choice(safe_moves) if safe_moves else random.randint(0, 3)
    else:
        if destination_station:
            if current_pos == destination_station:
                CARRYING = False
                passenger_station = None
                destination_station = None
                visited_stations.clear()  # Reset visited after drop-off
                return 5  # DROPOFF
            else:
                return move_toward(taxi_row, taxi_col, destination_station, obstacle_flags)
        else:
            unvisited_stations = [s for s in stations if s not in visited_stations]
            if unvisited_stations:
                nearest_station = min(unvisited_stations, key=lambda s: abs(taxi_row - s[0]) + abs(taxi_col - s[1]))
                return move_toward(taxi_row, taxi_col, nearest_station, obstacle_flags)
            else:
                # All stations visited, move randomly safely
                safe_moves = [a for a, blocked in zip([0,1,2,3], obstacle_flags) if not blocked]
                return random.choice(safe_moves) if safe_moves else random.randint(0, 3)
