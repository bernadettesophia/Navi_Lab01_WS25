# libraries
import numpy as np
import pandas as pd
import math as m
import time


# import files
path_data = r"C:\Users\berna\Documents\Studium\GST_2025W\Navigationssysteme_VU\NAVI_Labs\data"
### anpassen für os.


### einlesen mit pandas

# Nodes coordinates
nodepl = pd.read_csv(path_data + "\\nodepl.txt", delim_whitespace=True, 
                     skiprows=0, header=None,
                     names=["lat", "lon"])

# Nodes
nodelist = pd.read_csv(path_data + "\\nodelist.txt", delim_whitespace=True, 
                       skiprows=0, header=None,
                       names=["outgoing_arcs"])  # Index = ID

# Arcs mit Kanten
arclist = pd.read_csv(path_data + "\\arclist.txt", delim_whitespace=True, 
                      skiprows=0, header=None,
                      names=["nodes_ID", "time", "distance", "speed_limit", "clazz", "flags"])

### Mergiiiiing
# --- Schritt 0: Nodepl vorbereiten ---
nodepl_indexed = nodepl.copy()
nodepl_indexed['node_id'] = nodepl_indexed.index + 1  # Node IDs beginnen bei 1
nodepl_indexed.set_index('node_id', inplace=True)

# --- Schritt 1: Startknoten-Spalte in arclist erstellen ---
nodelist_valid = nodelist.iloc[:-1]
arc_to_startnode = np.zeros(len(arclist), dtype=int)
for i in range(len(nodelist_valid) - 1):
    start_idx = nodelist_valid.iloc[i, 0] - 1
    end_idx = nodelist_valid.iloc[i + 1, 0] - 1
    arc_to_startnode[start_idx:end_idx] = i + 1
arc_to_startnode[end_idx:] = len(nodelist_valid)
assert len(arc_to_startnode) == len(arclist), "Arc-Zuordnung passt nicht!"

arclist['from_node'] = arc_to_startnode
arclist['to_node'] = arclist['nodes_ID']

# --- Schritt 2: Start- und Zielknoten-Koordinaten mergen ---
df_merge_from = arclist.merge(
    nodepl_indexed, left_on='from_node', right_index=True, how='left'
).rename(columns={'lat':'lat_from','lon':'lon_from'})

df_nodes_to = nodepl_indexed.rename(columns={'lat':'lat_to','lon':'lon_to'})
df_supermerge = df_merge_from.merge(
    df_nodes_to, left_on='to_node', right_index=True, how='left'
)

# --- Graphen für alle Kostenfunktionen bauen ---
cost_functions = ['distance', 'time']
graph_dicts = {}
def build_graph(df, cost_col):
    graph = {}
    for _, row in df.iterrows():
        f = int(row['from_node'])
        t = int(row['to_node'])
        w = float(row[cost_col])
        if f not in graph:
            graph[f] = {}
        graph[f][t] = w
    for node_id in df['to_node'].unique():
        if node_id not in graph:
            graph[node_id] = {}
    return graph

for cost_col in cost_functions:
    graph_dicts[cost_col] = build_graph(df_supermerge, cost_col)


# Startknoten bestimmen (Home)
#lat_home = 47.079653893334836   # humboldt
#lon_home = 15.443076099512647

#lat_home = 47.07150627200452        # annen
#lon_home = 15.427614251514306

#lat_home = 47.02381186145014        # nippel
#lon_home = 15.426467146464514

lat_home = 47.05474875321875       # hecke
lon_home = 15.43991013295609


all_nodes = df_supermerge['from_node'].unique()
distances = ((nodepl_indexed.loc[all_nodes, 'lat'] - lat_home)**2 +
             (nodepl_indexed.loc[all_nodes, 'lon'] - lon_home)**2)
start_node_bike = distances.idxmin()
coord_start = nodepl_indexed.loc[start_node_bike]
print(f"Startknoten für Home: {start_node_bike}")
print(f"Koordinaten: lat={coord_start['lat']}, lon={coord_start['lon']}")



# Dijkstra-Algorithmus
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

def reconstruct_path(p_j, start, target):
    path = [target]
    while target in p_j:
        target = p_j[target]
        path.insert(0, target)
    if path[0] != start:
        return None
    return path

# Zielknoten-IDs extrahieren
targets = {
    "Basilika Mariatrost": 9328,
    "Schloss Eggenberg": 6031,
    "Murpark": 8543
}

# Koordinaten extrahieren
for name, node_id in targets.items():
    coords = nodepl.loc[node_id - 1]
    print(f"{name} (Node {node_id}): lat={coords['lat']:.6f}, lon={coords['lon']:.6f}")




# durchführen -> Pfade berechnen und für Export vorbereiten
start_time = time.time()

export_routes = []

for cost_col in cost_functions:
    graph = graph_dicts[cost_col]
    
    l_j, p_j = dijkstra(graph, start_node_bike)

    for name, node_id in targets.items():
        path = reconstruct_path(p_j, start_node_bike, node_id)
        if path is None:
            continue
        for idx, node in enumerate(path):
            export_routes.append({
                'target': name,
                'cost_function': cost_col,
                'seq': idx,
                'node_id': node,
                'lat': nodepl_indexed.loc[node, 'lat'],
                'lon': nodepl_indexed.loc[node, 'lon']
            })

end_time = time.time()
print(f"Dijkstra Rechenzeit: {end_time - start_time:.4f} Sekunden")

# In DataFrame umwandeln und exportieren
df_export = pd.DataFrame(export_routes)
#path_export = r"C:\Users\berna\Documents\Studium\GST_2025W\Navigationssysteme_VU\NAVI_Labs\results"
df_export.to_csv("dijkstra_routes.csv")
print("Export abgeschlossen: dijkstra_routes.csv")






# ----------------------------------------
# A* Algorithmus (nur distance, Export für QGIS)
# ----------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Berechnet die Luftliniendistanz zwischen zwei Punkten in km (statt m)."""
    R = 6371  # Erdradius in Metern
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def a_star_distance(graph, start, goal, node_coords):
    """A* für DISTANCE basierend auf g + h (beides Meter)."""

    g = {v: float('inf') for v in graph}
    f = {v: float('inf') for v in graph}
    p = {v: None for v in graph}

    g[start] = 0
    f[start] = haversine(
        node_coords.loc[start, 'lat'], node_coords.loc[start, 'lon'],
        node_coords.loc[goal, 'lat'],  node_coords.loc[goal, 'lon']
    )

    open_set = {start}
    closed_set = set()

    while open_set:
        # Knoten mit minimalem f wählen
        vi = min(open_set, key=lambda v: f[v])

        # Ziel erreicht?
        if vi == goal:
            break

        open_set.remove(vi)
        closed_set.add(vi)

        for vj, cij in graph[vi].items():
            if vj in closed_set:
                continue

            tentative_g = g[vi] + cij

            if tentative_g < g[vj]:
                p[vj] = vi
                g[vj] = tentative_g

                # Heuristik: reine Luftlinie (Meter)
                h = haversine(
                    node_coords.loc[vj, 'lat'], node_coords.loc[vj, 'lon'],
                    node_coords.loc[goal, 'lat'],  node_coords.loc[goal, 'lon']
                )
                f[vj] = g[vj] + h

                open_set.add(vj)

    # Pfad rekonstruieren
    node = goal
    if p[node] is None and node != start:
        return None, float('inf')

    path = []
    while node is not None:
        path.insert(0, node)
        node = p[node]

    return path, g[goal]



# ----------------------------------------
# A* nur für DISTANCE ausführen und exportieren
# ----------------------------------------

graph = graph_dicts["distance"]

start_time = time.time()

astar_routes = []

for name, node_id in targets.items():
    path, total_cost = a_star_distance(graph, start_node_bike, node_id, nodepl_indexed)
    if path is None:
        continue
    for idx, node in enumerate(path):
        astar_routes.append({
            'target': name,
            'cost_function': "distance_A*",
            'seq': idx,
            'node_id': node,
            'lat': nodepl_indexed.loc[node, 'lat'],
            'lon': nodepl_indexed.loc[node, 'lon']
        })

end_time = time.time()
print(f"A* Rechenzeit: {end_time - start_time:.4f} Sekunden")

df_astar = pd.DataFrame(astar_routes)
df_astar.to_csv("astar_routes_distance.csv")
#df_astar.to_csv(path_export + "\\routes_astar_distance_for_qgis.csv", index=False)
print("Export abgeschlossen: astar_routes_distance.csv")





# Sources:
# https://www.statology.org/pandas-find-closest-value/ -> find nearest node to home coordinates
# https://www.geeksforgeeks.org/python/python-measure-time-taken-by-program-to-execute/ -> time-mdulde for measure time taken by program to execute





