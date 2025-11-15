# Navigation Systems, Lab01, WS2015/26

### GST330UF Geospatial Technologies
### created by Bernadette Kakuska, 12109056 and Hannah Öttl, 11812239

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

## Heuristic used for A*

The A* implementation uses a geographical distance heuristic based on the Euclidean distance between the node coordinates (φ, λ). Since the dataset provides distances in kilometers, the heuristic is also computed in kilometers to ensure consistent units throughout the algorithm.

## Approximate Runtimes

- Dijkstra (six routes):  6.4454 seconds
- A* (three routes):  0.1589 seconds
- Full program (all routes, both algorithms): ~ 8.5 seconds

(Values may differ depending on hardware.)

## Potential Errors / Exceptions

- FileNotFoundError: Input files missing or wrong path.
- Runtime / Numerical Errors:
    - Floating-point differences: Small variations may occur when running on different machines.

## Code structure

1. Imports & Setup – Standard libraries (os, time, math) and third-party (numpy, pandas); input/output paths defined.

2. Data Loading – Reads nodepl.txt, nodelist.txt, and arclist.txt; merges node and arc data.

3. Graph Construction – Builds adjacency dictionaries for distance and time costs (graph_dicts).

4. Start Node Selection – Finds the closest node to the home coordinates.

5. Routing Algorithms – Implements Dijkstra and A* (with km-based heuristic).

6. Path Reconstruction & Metrics – Reconstructs paths, computes total distance and number of arcs.

7. Export & Visualization – Saves results to CSV files (results/) with node info, costs, and path sequence.