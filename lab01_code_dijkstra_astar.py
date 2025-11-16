"""
Created by: Hannah Öttl and Bernadette Kakuska
Date: 15.11.2025
Course: Navigation Systems (WS2025, GST330UF)
"""

# import libraries
import numpy as np
import pandas as pd
import math as m
import time
import os
import matplotlib.pyplot as plt

start_time_total = time.time()

# ----------------------------------------
# Data import (with pandas)
# ----------------------------------------

# file paths
path_data = os.path.join(os.path.dirname(__file__), "data")
path_results = os.path.join(os.path.dirname(__file__), "results")

# Node coordinates (geographical coordinates)
nodepl = pd.read_csv(path_data + "\\nodepl.txt", sep="\s+", 
                     skiprows=0, header=None,
                     names=["lat", "lon"])

# Nodelist
nodelist = pd.read_csv(path_data + "\\nodelist.txt", sep="\s+", 
                       skiprows=0, header=None,
                       names=["outgoing_arcs"])

# Arclist
arclist = pd.read_csv(path_data + "\\arclist.txt", sep="\s+", 
                      skiprows=0, header=None,
                      names=["to_node", "time", "distance", "speed_limit", "clazz", "flags"])


# ----------------------------------------
# Data preperations
# ----------------------------------------

# Copy node coordinates, assign 1-based node IDs (python starts with 0), 
# and set as index for easy access
nodepl_indexed = nodepl.copy()
nodepl_indexed['node_id'] = nodepl_indexed.index + 1
nodepl_indexed.set_index('node_id', inplace=True)

# make a copy and remove last “imaginary” node 
nodelist_clean = nodelist.iloc[:-1].copy()


# prepare array to store start nodes for each arc
arc_to_startnode = np.zeros(len(arclist), dtype=int)

# assign start node IDs based on cumulative outgoing arcs
for i in range(len(nodelist_clean) - 1):
    start_idx = nodelist_clean.iloc[i, 0] - 1       # first arc index of node i
    end_idx = nodelist_clean.iloc[i + 1, 0] - 1     # first arc index of next node
    arc_to_startnode[start_idx:end_idx] = i + 1     # node IDs start at 1

# assign remaining arcs to last valid node
arc_to_startnode[end_idx:] = len(nodelist_clean)

# assign to arclist
arclist['from_node'] = arc_to_startnode

# Merge start node coordinates
df_start_n_coord = arclist.merge(
    nodepl_indexed, left_on='from_node', 
    right_index=True, how='left').rename(columns={'lat':'lat_from','lon':'lon_from'})

# Rename end node coordinates
df_end_n_coord  = nodepl_indexed.rename(columns={'lat':'lat_to','lon':'lon_to'})

# Merge start and end coordinates into arcs
df_arc_coord  = df_start_n_coord.merge(
    df_end_n_coord, left_on='to_node', 
    right_index=True, how='left')

# Overwrite with only minimum-distance arcs per pair:
df_arc_coord = df_arc_coord.loc[
    df_arc_coord.groupby(['from_node', 'to_node'])['distance'].idxmin()
].reset_index(drop=True)


# Define home location (Heckengasse)
lat_home = 47.05474875321875
lon_home = 15.43991013295609

# Find the closest network node to home
all_nodes = df_arc_coord['from_node'].unique()
distances = ((nodepl_indexed.loc[all_nodes, 'lat'] - lat_home) ** 2 +
             (nodepl_indexed.loc[all_nodes, 'lon'] - lon_home) ** 2)

# Take the node with the smallest distance as the starting point
start_node = distances.idxmin()
coord_start = nodepl_indexed.loc[start_node]

# ----------------------------------------
# Print data summary
# ----------------------------------------
num_nodes = len(nodepl_indexed)
num_arcs = len(df_arc_coord)
print(f"Nodes: {num_nodes} \nArcs: {num_arcs}")
print(f"Start node: {start_node} (lat={coord_start['lat']:.6f}, lon={coord_start['lon']:.6f})\n")

# ----------------------------------------
# Building Graphs for Routing Algorithms
# ----------------------------------------

# Build graphs for all cost metrics
cost_functions = ['distance', 'time']  # Define which metrics to use for graph weights
graph_dicts = {}                        # Dictionary to store graphs per cost function

################
def build_graph(df, cost_col):
    graph = {}  # Initialize empty dictionary for the graph

    # Iterate over each arc (row) in the DataFrame
    for _, row in df.iterrows():
        from_node = int(row['from_node'])
        to_node = int(row['to_node'])
        weight = float(row[cost_col])       # Use the selected cost as edge weight
        
        # If the start node is not in the graph yet, add it
        if from_node not in graph:
            graph[from_node] = {}
        
        # Add the edge with its weight, keep minimum if multiple arcs exist
        if to_node not in graph[from_node] or weight < graph[from_node][to_node]:
            graph[from_node][to_node] = weight
    
    # Ensure all nodes exist in the graph (even nodes with no outgoing arcs)
    for node_id in df['to_node'].unique():
        if node_id not in graph:
            graph[node_id] = {}

    return graph

# Build a separate graph for each cost function
for cost in cost_functions:
    graph_dicts[cost] = build_graph(df_arc_coord, cost)
print(f"Graphs built for costs: {', '.join(cost_functions)}\n")

# Extract target nodes
targets = {
    "Basilika Mariatrost": 9328,
    "Schloss Eggenberg": 6031,
    "Murpark": 8543
}

# ----------------------------------------
# Compute shortest paths using Dijkstras algorithm
# ----------------------------------------

# Dijkstra algorithm (based on pseudo code on course slides)
def dijkstra(graph, v_s):
    V = set(graph.keys())       # V -> all nodes
    T = {v_s}               # T -> set of nodes with a temporary label
    P = set()               #  set of nodes with a permanent label

    l_j = {v: m.inf for v in V}         # Initialize distances: l_j = inf for all j
    p_j = {v: None for v in V}          # p_j -> Predecessor of a node v_j
    l_j[v_s] = 0            # Start node label

    while T:
        v_i = min(T, key=lambda v: l_j[v])      # Pick node in T with smallest label
        T.remove(v_i)       # Move v_i from T to P
        P.add(v_i)

        # For each neighbor
        for v_j, c_ij in graph[v_i].items():

            # If v_j not in T and not in P -> add v_j to T
            if v_j not in T and v_j not in P:
                l_j[v_j] = l_j[v_i] + c_ij
                p_j[v_j] = v_i
                T.add(v_j)

            # If v_j already in T and new label is smaller -> update label
            elif v_j in T and l_j[v_i] + c_ij < l_j[v_j]:
                l_j[v_j] = l_j[v_i] + c_ij
                p_j[v_j] = v_i

            # If v_j is in P AND new label is smaller
            elif v_j in P and l_j[v_i] + c_ij < l_j[v_j]:
                # move node back to T
                l_j[v_j] = l_j[v_i] + c_ij
                p_j[v_j] = v_i
                P.remove(v_j)
                T.add(v_j)

    # return visited nodes set
    num_visited_nodes = len(P)
    return l_j, p_j, num_visited_nodes, P

# Reconstruct the path from start to target using the predecessors
def reconstruct_path(p_j, start, target):
    path = []
    current = target

    # Stop if no predecessor exists
    if p_j[current] is None and current != start:
        return None

    # go backwards through the predecessor list
    while current is not None:
        path.insert(0, current)     # add the current node at the beginning of the list
        if current == start:
            return path
        current = p_j[current]

    return None

# ----------------------------------------
# Run Dijkstra (mit Zeit pro Ziel)
# ----------------------------------------
dijkstra_expanded = {cost: {} for cost in cost_functions}
a_star_expanded = {}

# sets that collect ALL expanded nodes across all targets (for plotting)
expanded_n_dij_dist = set()
expanded_n_a_dist = set()

for cost in cost_functions:     # Loop over each cost function (distance/time)
    print(f"Running Dijkstra for cost: {cost}")
    start_time_total_cost = time.time()
    
    l_j_dummy, p_j_dummy, expanded_nodes_dummy, visited_set_dummy = dijkstra(graph_dicts[cost], start_node)
    
    elapsed_total = time.time() - start_time_total_cost
    print(f"  Dijkstra total time for all targets: {elapsed_total:.4f} s, expanded nodes: {expanded_nodes_dummy}")

    for name, node_id in targets.items():
        t0 = time.time()  # Timer per route

        l_j, p_j, expanded_nodes, visited_set = dijkstra(graph_dicts[cost], start_node)
        dijkstra_expanded[cost][name] = visited_set

        # collect only DISTANCE expansions for fair comparison
        if cost == "distance":
            expanded_n_dij_dist.update(visited_set)

        path = reconstruct_path(p_j, start_node, node_id)
        t_elapsed = time.time() - t0

        if path is None:
            continue

        if cost == "distance":
            total_distance = sum(
                df_arc_coord.loc[
                    (df_arc_coord['from_node'] == path[i]) &
                    (df_arc_coord['to_node'] == path[i+1]),
                    'distance'
                ].values[0]
                for i in range(len(path)-1)
            )
            print(f"  Dijkstra to {name}: time per route {t_elapsed:.6f}s, expanded nodes: {expanded_nodes}, path length: {len(path)}, total distance: {total_distance:.2f} km")
        elif cost == "time":
            total_time = sum(
                df_arc_coord.loc[
                    (df_arc_coord['from_node'] == path[i]) &
                    (df_arc_coord['to_node'] == path[i+1]),
                    'time'
                ].values[0]
                for i in range(len(path)-1)
            )
            print(f"  Dijkstra to {name}: time per route {t_elapsed:.6f}s, expanded nodes: {expanded_nodes}, path length: {len(path)}, total time: {total_time:.2f} h")
    print()


# ----------------------------------------
# Compute shortest paths using A* Algorithm
# ----------------------------------------

# Calculate straight-line distance between two points in km using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# A* algorithm (based on pseudo code from course slides)
def a_star(graph, start, desti, node_coords):

    l_j = {v: float('inf') for v in graph}   # l_j: real cost from start to node j
    f_j = {v: float('inf') for v in graph}   # f_j = l_j + h_j (total evaluation)
    p_j = {v: None for v in graph}           # p_j: predecessor of node j

    # start node gets cost 0
    l_j[start] = 0

    # f_s = h_s (since g_s = 0)
    f_j[start] = haversine(
        node_coords.loc[start, 'lat'], node_coords.loc[start, 'lon'],
        node_coords.loc[desti, 'lat'],  node_coords.loc[desti, 'lon']
    )

    T = {start}   # T: nodes with temporary label (to be explored)
    P = set()     # P: nodes with permanent label (already processed)

    # main loop
    while T:

        # pick node with smallest f_j (slide line 5)
        v_i = min(T, key=lambda v: f_j[v])

        # stop if destination is reached
        if v_i == desti:
            break

        # move v_i from T to P (slide line 6)
        T.remove(v_i)
        P.add(v_i)

        # check all neighbors (slide line 7)
        for v_j, c_i_j in graph[v_i].items():

            if v_j in P:     # skip permanently labeled nodes
                continue

            new_cost = l_j[v_i] + c_i_j   # l_i + c_ij

            # if this path is better -> update labels
            if new_cost < l_j[v_j]:
                l_j[v_j] = new_cost       # update l_j
                p_j[v_j] = v_i            # update p_j

                # compute heuristic h_j
                h_j = haversine(
                    node_coords.loc[v_j, 'lat'], node_coords.loc[v_j, 'lon'],
                    node_coords.loc[desti, 'lat'], node_coords.loc[desti, 'lon']
                )

                f_j[v_j] = l_j[v_j] + h_j    # update f_j

                T.add(v_j)   # add to T if not already there

    # reconstruct the path from predecessors (only for counting nodes)
    path = []
    current = desti

    if p_j[current] is None and current != start:
        return None, l_j[desti], len(P), P

    while current is not None:
        path.insert(0, current)
        if current == start:
            break
        current = p_j[current]

    return path, l_j[desti], len(P), P


# ----------------------------------------
# Run A* (distance heuristic) pro Ziel
# ----------------------------------------
print(f"Running A* (distance heuristic) for each target")
graph = graph_dicts["distance"]
start_time_all = time.time()

for name, node_id in targets.items():
    t0 = time.time()  # Timer for this route

    path, total_cost, expanded_nodes, visited_set = a_star(graph, start_node, node_id, nodepl_indexed)

    expanded_n_a_dist.update(visited_set)

    t_elapsed = time.time() - t0  # time per target
    if path is None:
        continue

    # only distance is relevant here
    total_distance = sum(
        df_arc_coord.loc[
            (df_arc_coord['from_node'] == path[i]) &
            (df_arc_coord['to_node'] == path[i+1]),
            'distance'
        ].values[0]
        for i in range(len(path)-1)
    )

    print(f"  A* to {name}: time per route {t_elapsed:.6f}s, expanded nodes: {expanded_nodes}, path length: {len(path)}, total distance: {total_distance:.2f} km")

total_a_star_time = time.time() - start_time_all
print(f"\nTotal A* time for all targets: {total_a_star_time:.4f}s\n")




# ----------------------------------------
# Visualization: per-cost-function comparison for shortest distance
# ----------------------------------------
dij_exp_set = expanded_n_dij_dist - expanded_n_a_dist
astar_exp_set = expanded_n_a_dist

plt.figure(figsize=(12,8))

# Dijkstra nodes (distance)
plt.scatter(nodepl_indexed.loc[list(dij_exp_set), 'lon'],
            nodepl_indexed.loc[list(dij_exp_set), 'lat'],
            color='#c27e16', s=10, label='Dijkstra')

# A* nodes (distance)
plt.scatter(nodepl_indexed.loc[list(astar_exp_set), 'lon'],
            nodepl_indexed.loc[list(astar_exp_set), 'lat'],
            color='#9a1d1d', s=10, label=f'A* (n={len(dij_exp_set)})')

# Start node
plt.scatter(nodepl_indexed.loc[start_node, 'lon'],
            nodepl_indexed.loc[start_node, 'lat'],
            color="#312007", s=50, marker='v', label='Start Node')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Comparison of Expanded Nodes: Dijkstra vs A* (Distance Cost)")
plt.legend()
plt.axis('equal')

# Save the figure
os.makedirs(path_results, exist_ok=True)
file_path = os.path.join(path_results, "expanded_nodes_comparison.png")
plt.savefig(file_path, dpi=300)

plt.show()
plt.close()


# ----------------------------------------
# Visualization: per-cost-function comparison for shortest distance to "Murpark"
# ----------------------------------------

# Extract only the expanded nodes for “Murpark”
target_name = "Murpark"
target_node = targets[target_name]

# Run Dijkstra for just Murpark
_, _, _, expanded_dij_murpark = dijkstra(graph_dicts["distance"], start_node)

# Run A* for just Murpark
_, _, _, expanded_a_murpark = a_star(graph_dicts["distance"], start_node, target_node, nodepl_indexed)

# Compute sets for plotting
dij_exp_set_murpark = expanded_dij_murpark - expanded_a_murpark
astar_exp_set_murpark = expanded_a_murpark

plt.figure(figsize=(12,8))

# Dijkstra nodes
plt.scatter(nodepl_indexed.loc[list(dij_exp_set_murpark), 'lon'],
            nodepl_indexed.loc[list(dij_exp_set_murpark), 'lat'],
            color='#c27e16', s=10, label=f'Dijkstra (n={len(dij_exp_set_murpark)})')

# A* nodes
plt.scatter(nodepl_indexed.loc[list(astar_exp_set_murpark), 'lon'],
            nodepl_indexed.loc[list(astar_exp_set_murpark), 'lat'],
            color='#9a1d1d', s=10, label=f'A* (n={len(astar_exp_set_murpark)})')

# Start node
plt.scatter(nodepl_indexed.loc[start_node, 'lon'],
            nodepl_indexed.loc[start_node, 'lat'],
            color="#312007", s=50, marker='v', label='Start Node')

# Target node (Murpark)
plt.scatter(nodepl_indexed.loc[target_node, 'lon'],
            nodepl_indexed.loc[target_node, 'lat'],
            color='green', s=50, marker='X', label='Target: Murpark')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Expanded Nodes to Target: Murpark — Dijkstra vs A* (Distance Cost)")
plt.legend()
plt.axis('equal')

file_path = os.path.join(path_results, "expanded_nodes_murpark.png")
plt.savefig(file_path, dpi=300)

plt.show()
plt.close()




