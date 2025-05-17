
# 🚦 Smart Infrastructure Management GUI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

A modular and scalable Python-based application for analyzing and optimizing urban road and transit networks. This GUI-enabled tool empowers urban planners, transport engineers, and researchers to simulate, visualize, and enhance infrastructure using advanced algorithms like Minimum Spanning Tree (MST), Dijkstra’s, A\*, and dynamic programming.

---

## 🚀 Core Features

| Category         | Features                                                            |
| ---------------- | ------------------------------------------------------------------- |
| 💻 GUI           | Tkinter-based interactive simulation environment                    |
| 📊 Algorithms    | MST (Kruskal), Dijkstra, A\* (with heuristics), Dynamic Programming |
| 🧠 Optimization  | Bus scheduling, road maintenance, metro line analysis               |
| 🧭 Routing       | Traffic-aware pathfinding (rush hour simulation)                    |
| 🎨 Visualization | Real-time network rendering with Matplotlib & NetworkX              |
| 🧪 Testing       | Unit-tested algorithm modules using Python’s unittest framework     |

---

## 🖼️ Demo Preview (optional)

> Insert screenshots or gifs here, for example:
>
> * Main GUI interface
> * Graph visualizations
> * Bus route optimization output
> * MST overlays on map

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Smart-Infrastructure-GUI.git
cd Smart-Infrastructure-GUI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:

```
matplotlib
networkx
pandas
```

---

## ▶️ Run the Application

You can launch the GUI using:

```bash
python InfrastructureEngineerApp.py
```

Alternative entry point (transit-focused GUI):

```bash
python Algorithms\ project\ with\ GUI.py
```

---

## 🧠 Algorithms Implemented

* 🟩 Kruskal’s MST for minimum-cost infrastructure layout
* 🟦 Dijkstra’s Algorithm for shortest route planning
* 🟨 A\* Search for emergency and heuristic-driven routing
* 🟥 Time-Dependent Shortest Path for congestion-aware navigation
* 🟧 Dynamic Programming for:

  * 🚌 Optimal bus coverage within budget
  * 🛠️ Priority-based road maintenance
* 🟪 Greedy Algorithms for:

  * 🚦 Traffic signal optimization
  * 🚑 Emergency vehicle preemption

---

## 🗂️ Project Structure

```
.
├── InfrastructureEngineerApp.py        # Main simulation GUI
├── Algorithms project with GUI.py      # Transit-optimized GUI
├── RoadNetwork.py                      # Graph structure + all algorithms
├── RoadNetworkVisualizer.py            # Visualization module
├── UnionFind.py                        # Kruskal's helper class
├── TestRoadNetworkAnalysis.py          # Unit test coverage
├── requirements.txt                    # Required packages
```

---

## ✅ Testing

To validate algorithm correctness and edge cases:

```bash
python -m unittest TestRoadNetworkAnalysis.py
```

Tests include:

* MST validation
* Shortest path consistency (Dijkstra vs A\*)
* Redundant edge handling
* Edge case scenarios (disconnected graphs, missing nodes)

---

## 📈 Use Cases

✔️ Plan cost-effective road expansions
✔️ Simulate and compare traffic routing under peak congestion
✔️ Optimize bus network to maximize coverage
✔️ Prioritize infrastructure maintenance with resource constraints
✔️ Rank metro line usage based on simulated passenger load

---

## 🌐 Future Enhancements

* 🔄 Live traffic data integration (e.g., Google Maps API)
* 🧠 Machine Learning for congestion prediction
* 🤖 RL-based traffic light control
* 🗺️ GIS visualization with Folium/Leaflet
* 🌍 Web-based dashboard using Flask or FastAPI

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

