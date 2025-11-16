# Navigation Systems, Lab01, WS2025/26, GST330UF

#### Authors: Kakuska, B. & & Öttl, H.
#### Date: 16.11.2025

## Project description

This project computes optimal routes in Graz using Dijkstra’s algorithm and the A* algorithm. Time- and distance-based paths to three destinations are calculated, visualized, and compared to evaluate the performance and differences between both routing methods.

## Requirements

- Programming language: Python 3.11.4
- Packages:
    - numpy: 1.26.3
    - pandas: 2.1.
- Input files (must be in the same folder):
    - nodepl.txt
    - nodelist.txt
    - arclist.txt


## Potential Errors / Exceptions

- FileNotFoundError:
    - Input files missing or wrong path.
- Runtime / Numerical Errors:
    - Floating-point differences: Small variations may occur when running on different machines.

## Code structure

1. Imports & Setup – Libraries (os, time, math, numpy, pandas, matplotlib) are imported; input/output paths are defined.

2. Data Loading & Preprocessing – Reads nodepl.txt, nodelist.txt, and arclist.txt; assigns node IDs, removes dummy nodes, merges start/end coordinates, and keeps only minimum-distance arcs per node pair.

3. Start Node Selection – Determines the closest network node to the home location using coordinate comparison.

4. Graph Construction – Builds adjacency dictionaries for each cost metric (distance and time) for use in routing algorithms.

5. Routing Algorithms – Implements Dijkstra and A* algorithms. A* uses the Haversine formula as a straight-line distance heuristic.

6. Path Reconstruction & Metrics – Reconstructs shortest paths, computes total distance, counts expanded nodes, and tracks visited nodes for visualization.

7. Visualization & Export – Plots expanded nodes for Dijkstra vs A*, including full network and single-target comparisons; saves results and figures to results/.

## Heuristic used for A*

The A* implementation uses a geographical distance heuristic based on the Euclidean distance between the node coordinates (φ, λ). Since the dataset provides distances in kilometers, the heuristic is also computed in kilometers to ensure consistent units throughout the algorithm.

## Approximate Runtimes

- Dijkstra (Basilika Mariatrost): 0.136564 seconds
- Dijkstra (Murpark):0.120757 seconds
- Dijkstra (Schloss Eggenberg): 0.125244 seconds
- A* (Basilika Mariatrost): 0.095032 seconds
- A* (Murpark): 0.015579 seconds
- A* (Schloss Eggenberg): 0.048208 seconds
- Full program (all routes, both algorithms): ~ 0.3798 seconds

(Values may differ depending on hardware.)

## Results and Output Explanation

The program generates several output files and plots in the 'results' folder for analysis and visualization:

1. Expanded Nodes Comparison (expanded_nodes_comparison.png)

    - Shows all nodes expanded by Dijkstra and A* algorithms for the distance cost.

    - Nodes expanded only by Dijkstra are marked in orange, nodes expanded only by A* in red, and nodes expanded by both algorithms in purple.

    - The start node is highlighted with a black marker.

2. Target-Specific Expanded Nodes (expanded_nodes_murpark.png)

    - Visualizes nodes expanded when calculating the path specifically to the Murpark target.

    - Compares Dijkstra and A* expansions, highlighting the start node and the target node.

    - Helps illustrate efficiency differences between the algorithms for a single route.

3. CSV/Other Data 

    - The program can save computed paths, total distances, and expanded node sets for further analysis if implemented.

## Sources

- Chand, S. (2021). Here’s how to calculate distance between 2 geolocations in Python. Towards Data Science. Retrieved November 16, 2025, from https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4/ -> Haversine function for A* Heuristic

- Course slides: "02a_Routing.pdf", p.29. -> Pseudocode for Dijkstra's algorithm taken from course slides

- Course slides: "02a_Routing.pdf", p.50. -> Pseudocode for A* algorithm taken from course slides

- GeeksforGeeks. (2025). Measure time taken by program to execute in Python. Retrieved November 16, 2025, from https://www.geeksforgeeks.org/python/python-measure-time-taken-by-program-to-execute/ -> time-module to measure time taken by program to execute

- Hunter, J. D., et al. (2025). Matplotlib — Visualization with Python. Retrieved November 16, 2025, from https://matplotlib.org/stable/index.html ->for creating plots for visualisation

- OpenAI. (2025). ChatGPT (Version GPT-5-mini) [Large language model]. Retrieved November 16, 2025, from https://openai.com/ -> used for code debugging, error detection, code cleanup, and suggestions for implementation improvements

- pandas development team. (2025). pandas.read_csv — pandas documentation. Retrieved November 16, 2025, from https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    -> import files with pandas

- Perplexity. (2025). Perplexity AI platform. Retrieved November 16, 2025, from https://www.perplexity.ai/ -> used for code debugging, error detection, code cleanup, and suggestions for implementation improvements

- Statology. (2022). How to Find Closest Value in Pandas DataFrame (With Example). Retrieved November 16, 2025, from https://www.statology.org/pandas-find-closest-value/ -> find nearest node to home coordinates

