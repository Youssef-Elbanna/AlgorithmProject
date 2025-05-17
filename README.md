

# Smart Infrastructure Management GUI

A Python-based GUI application for infrastructure network analysis and optimization. This tool enables urban planners and civil engineers to visualize, analyze, and optimize road networks using algorithms like Minimum Spanning Tree (MST), Dijkstra, and A\* with traffic-awareness.

## Features

* **Interactive Tkinter GUI**
* **Graph-based road network management**
* **Minimum Spanning Tree (MST) generation**
* **Shortest path finding with Dijkstra and A\***
* **Traffic-aware route analysis**
* **Greedy transit optimization**
* **Custom visualization using `matplotlib` and `networkx`**
* **Transit analysis features:**

  * Optimal bus scheduling
  * Road maintenance priority planning
  * Metro line usage ranking

## Installation

### Requirements

Install dependencies from `requiremtnt.txt`:

```bash
pip install -r requiremtnt.txt
```

Contents of `requiremtnt.txt`:

```
matplotlib
networkx
pandas
```

### Run the App

```bash
python InfrastructureEngineerApp.py
```

## Project Structure

```
.
├── InfrastructureEngineerApp.py    # Main GUI application
├── RoadNetwork.py                  # Core graph and algorithm logic
├── RoadNetworkVisualizer.py        # Graph visualization utilities
├── Algorithm_project_with_GUI.py   # Transit optimization functions
├── UnionFind.py                    # Union-Find class for MST
├── TestRoadNetworkAnalysis.py      # Unit tests for RoadNetwork
├── requiremtnt.txt                 # Project dependencies
```

## Testing

Run unit tests using:

```bash
python -m unittest TestRoadNetworkAnalysis.py
```

## Example Use Cases

* Determine the **most cost-effective road layout** using MST.
* Find **fastest or most reliable routes** under current traffic conditions.
* Analyze **bus stop coverage** and schedule efficiency.
* Prioritize **road repairs** based on connectivity importance.
* Evaluate and rank **metro line usage** for public transport planning.



