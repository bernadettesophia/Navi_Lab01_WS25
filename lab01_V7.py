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
        
        # Add the edge with its weight
        graph[from_node][to_node] = weight
    
    # Ensure all nodes exist in the graph (even nodes with no outgoing arcs)
    for node_id in df['to_node'].unique():
        if node_id not in graph:
            graph[node_id] = {}

    return graph

# Build a separate graph for each cost function
for cost in cost_functions:
    graph_dicts[cost] = build_graph(df_arc_coord, cost)


# ----------------------------------------
# Compute shortest paths using Dijkstras algorithm
# ----------------------------------------

# Dijkstra algorithm (based on pseudo code on course slides)


###### variablen 


def dijkstra(graph, v_s): 
    T = set(graph.keys())
    l_j = {v: m.inf for v in graph}
    p_j = {}
    l_j[v_s] = 0
    while T:
        v_i = min(T, key=lambda v: l_j[v])
        if l_j[v_i] == m.inf:
            break
        T.remove(v_i)
        for v_j, c_ij in graph[v_i].items():
            if v_j in T:
                new_l = l_j[v_i] + c_ij
                if new_l < l_j[v_j]:
                    l_j[v_j] = new_l
                    p_j[v_j] = v_i
    return l_j, p_j

# Reconstruct the path from start to target using the predecessors
def reconstruct_path(p_j, start, target):
    path = [target]     # Start from the target
    while target in p_j:        # Trace back using predecessors
        target = p_j[target]        # Insert nodes at the beginning
        path.insert(0, target)      # If path doesn't start with start node
    if path[0] != start:        # No path exists
        return None
    return path

# Extract target nodes
targets = {
    "Basilika Mariatrost": 9328,
    "Schloss Eggenberg": 6031,
    "Murpark": 8543
}

# Execute function to compute shortest paths
start_time = time.time()        # start timer

export_routes = []

for cost in cost_functions:     # Loop over each cost function (distance/time)
    graph = graph_dicts[cost]       # Get the graph for the given cost metric
    
    l_j, p_j = dijkstra(graph, start_node)

    for name, node_id in targets.items():
        path = reconstruct_path(p_j, start_node, node_id)
        if path is None:
            continue
        for idx, node in enumerate(path):       # Loop over nodes in path
            export_routes.append({          # Node info for export
                'target': name,
                'cost_function': cost,
                'seq': idx,
                'node_id': node,
                'lat': nodepl_indexed.loc[node, 'lat'],
                'lon': nodepl_indexed.loc[node, 'lon']
            })

end_time = time.time()      # End timer
print(f"Dijkstra computation time: {end_time - start_time:.4f} seconds")

# Convert the results to a DataFrame and export results as .csv
df_dijkstra = pd.DataFrame(export_routes)
df_dijkstra.to_csv(os.path.join(path_results, "dijkstra_routes.csv"))


# ----------------------------------------
# Compute shortest paths using A* Algorithm 
# ----------------------------------------

# Calculate straight-line distance between two points in km using the Haversine formula         ###quelleeee
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# A* algorithm (based on pseudo code on course slides)
def a_star(graph, start, desti, node_coords):

    # initialize variables
    g = {v: float('inf') for v in graph}  # l_j: cost from start to node
    f = {v: float('inf') for v in graph}  # f_j: estimated total cost (g + h)
    p = {v: None for v in graph}          # p_j: predecessor node

    g[start] = 0
    f[start] = haversine(           # f_start = g_start + h_start (heuristic to desti)
        node_coords.loc[start, 'lat'], node_coords.loc[start, 'lon'],
        node_coords.loc[desti, 'lat'],  node_coords.loc[desti, 'lon']
    )

    T = {start}   # T: nodes to explore
    P = set()   # P: nodes already processed

    # --- Step 2: main loop ---
    while T:
        v_i = min(T, key=lambda v: f[v])        # Pick node with smallest f_j (like line 5)

        if v_i == desti:        # Stop if the destination is reached
            break

        T.remove(v_i)     # remove from T
        P.add(v_i)      # add to P

        # --- Step 3: examine neighbors ---
        for v_j, c_i_j in graph[v_i].items():  # for v_j in N(v_i)
            if v_j in P:  # skip already processed nodes
                continue

            tentative_g = g[v_i] + c_i_j  # l_i + c_ij

            # If new path is shorter (like lines 12 & 15)
            if tentative_g < g[v_j]:
                g[v_j] = tentative_g     # update l_j
                p[v_j] = v_i             # update p_j

                # f_j = g_j + h_j (line 10, 14, 17)
                h = haversine(
                    node_coords.loc[v_j, 'lat'], node_coords.loc[v_j, 'lon'],
                    node_coords.loc[desti, 'lat'], node_coords.loc[desti, 'lon']
                )
                f[v_j] = g[v_j] + h

                T.add(v_j)        # add v_j to T if not there yet

    # Reconstruct path from desti to start ---
    node = desti
    if p[node] is None and node != start:  # no path exists
        return None, float('inf')

    path = []
    while node is not None:
        path.insert(0, node)  # insert at beginning
        node = p[node]

    return path, g[desti]



# ----------------------------------------
# Run A* (only for distance)
# ----------------------------------------

graph = graph_dicts["distance"]  # use distance graph

start_time = time.time()          # start timer

a_star_routes = []

for name, node_id in targets.items():
    # find path and total cost
    path, total_cost = a_star(graph, start_node, node_id, nodepl_indexed)
    if path is None:
        continue   # skip if no path found

    # store path info for export
    for idx, node in enumerate(path):
        a_star_routes.append({
            'target': name,
            'cost_function': "distance_A*",
            'seq': idx,
            'node_id': node,
            'lat': nodepl_indexed.loc[node, 'lat'],
            'lon': nodepl_indexed.loc[node, 'lon']
        })

end_time = time.time()  # end timer
print(f"A* computation time: {end_time - start_time:.4f} seconds")

# Convert results to DataFrame and export
df_a_star = pd.DataFrame(a_star_routes)
df_a_star.to_csv(os.path.join(path_results, "astar_routes_distance.csv"))






### functions to calculate length and operations

def calculate_metrics(df, path, cost_col):
    """Berechnet die Summe der Kantengewichte und Anzahl der Kanten für einen Pfad."""
    length = 0.0
    for i in range(len(path)-1):
        edge = df[(df['from_node'] == path[i]) & (df['to_node'] == path[i+1])]
        length += edge[cost_col].values[0]
    operations_count = len(path) - 1
    return length, operations_count


def print_metrics(routes_df):
    grouped = routes_df.groupby(['target', 'cost_function'])
    for (target, cost_func), group in grouped:
        length = group['length'].iloc[0]
        operations = group['operations'].iloc[0]
        print(f"Ziel: {target}, Algorithmus: {cost_func}")
        print(f"  Pfadlänge: {length:.2f}")
        print(f"  Operationen (Kanten): {operations}")
        print(f"  Anzahl Knoten im Pfad: {len(group)}\n")


# ----------------------------------------
# Berechnung der Länge und Operationen
# ----------------------------------------

# Berechnung der Länge und Operationen für Dijkstra
for cost_col in cost_functions:
    df_subset = df_dijkstra[df_dijkstra['cost_function'] == cost_col]
    for target, group in df_subset.groupby('target'):
        path = group['node_id'].tolist()
        length, operations = calculate_metrics(df_arc_coord, path, cost_col)
        df_dijkstra.loc[
            (df_dijkstra['target'] == target) &
            (df_dijkstra['cost_function'] == cost_col),
            'length'
        ] = length
        df_dijkstra.loc[
            (df_dijkstra['target'] == target) &
            (df_dijkstra['cost_function'] == cost_col),
            'operations'
        ] = operations

# Berechnung für A*
for target, group in df_astar.groupby('target'):
    path = group['node_id'].tolist()
    length, operations = calculate_metrics(df_arc_coord, path, 'distance')
    df_astar.loc[df_astar['target'] == target, 'length'] = length
    df_astar.loc[df_astar['target'] == target, 'operations'] = operations

# Jetzt ist print_metrics erlaubt
print("=== Dijkstra Metriken ===")
print_metrics(df_dijkstra)

print("=== A* Metriken ===")
print_metrics(df_astar)


end_time_total = time.time()
print(f"Total Rechenzeit: {end_time_total - start_time_total:.4f} Sekunden")



# Sources:
# https://www.statology.org/pandas-find-closest-value/ -> find nearest node to home coordinates
# https://www.geeksforgeeks.org/python/python-measure-time-taken-by-program-to-execute/ -> time-module for measure time taken by program to execute
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html -> import files with pandas






