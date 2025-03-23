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

def move_toward(taxi_row, taxi_col, target, obstacle_flags):
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obstacle_flags
    target_row, target_col = target
    d_row = target_row - taxi_row
    d_col = target_col - taxi_col

    actions = []
    if d_row < 0 and not obstacle_north:
        actions.append(1)  # North
    if d_row > 0 and not obstacle_south:
        actions.append(0)  # South
    if d_col > 0 and not obstacle_east:
        actions.append(2)  # East
    if d_col < 0 and not obstacle_west:
        actions.append(3)  # West

    if actions:
        return random.choice(actions)

    safe_moves = [a for a, blocked in zip([0,1,2,3], obstacle_flags) if not blocked]
    return random.choice(safe_moves) if safe_moves else random.randint(0, 3)

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
