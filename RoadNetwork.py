# RoadNetwork.py
from collections import defaultdict
import heapq
import math
from UnionFind import UnionFind  # For Kruskal's algorithm

class RoadNetwork:
    def __init__(self):
        # Initialize node and edge structures
        self.nodes = {}  # node_id -> node_type
        self.node_metadata = {}  # node_id -> metadata like type, x, y, population
        self.edges = []  # list of tuples: (source, destination, distance, condition)
        self.adj_list = defaultdict(list)  # adjacency list for graph traversal
        self.critical_facilities = set()  # node_ids of critical facilities
        self.neighborhoods = set()  # node_ids of neighborhoods
        self.traffic_flow = {}  # edge -> traffic volume data by time periods

    def add_node(self, node_id, node_type='regular', x=None, y=None, population=0):
        # Add a node to the network
        self.nodes[node_id] = node_type
        self.node_metadata[node_id] = {
            'type': node_type,
            'x': x,
            'y': y,
            'population': population
        }

        # Classify node based on type
        t = node_type.lower()
        if t in ['critical', 'airport', 'education', 'medical', 'government', 'transit hub']:
            self.critical_facilities.add(node_id)
        elif t in ['neighborhood', 'residential', 'mixed']:
            self.neighborhoods.add(node_id)

    def add_edge(self, src, dest, distance, condition):
        # Add an edge and update the adjacency list
        self.edges.append((src, dest, distance, condition))
        self.adj_list[src].append((dest, condition))
        self.adj_list[dest].append((src, condition))

    def _adjust_weight(self, src, dest, base_weight):
        # Adjust edge weight based on critical facility and population biases
        weight = base_weight

        # Critical facility nodes have reduced weight
        if src in self.critical_facilities or dest in self.critical_facilities:
            weight *= 0.8

        # Apply population influence
        pop_src = self.node_metadata[src].get('population', 0)
        pop_dest = self.node_metadata[dest].get('population', 0)
        avg_pop = (pop_src + pop_dest) / 2
        pop_factor = 1 - min(avg_pop / 10000, 0.2)  # max 20% reduction
        weight *= pop_factor

        print(f"Edge {src} ↔ {dest}: base={base_weight:.2f}, pop_avg={avg_pop:.0f}, weight={weight:.2f}")  # Debug log

        return weight

    def kruskal_mst(self):
        # Generate Minimum Spanning Tree using Kruskal’s algorithm
        uf = UnionFind(self.nodes)
        # Use distance as the cost for MST, with critical bias
        weighted_edges = [
            (s, d, self._adjust_weight(s, d, dist))
            for s, d, dist, _ in self.edges
        ]
        weighted_edges.sort(key=lambda x: x[2])  # Sort by weight

        mst = []
        total_cost = 0
        for src, dest, w in weighted_edges:
            if uf.union(src, dest):
                mst.append((src, dest, w))
                total_cost += w
        return mst, total_cost

    def calculate_total_cost(self):
        # Estimate construction and maintenance cost
        construction = sum(dist for _, _, dist, _ in self.edges)
        maintenance = construction * 0.1
        return {
            'construction_cost': construction,
            'maintenance_cost': maintenance,
            'first_year_total': construction + maintenance
        }

    def calculate_connectivity_metrics(self):
        # Calculate general connectivity metrics
        n = len(self.nodes)
        return {
            'total_nodes': n,
            'total_edges': len(self.edges),
            'avg_degree': (sum(len(v) for v in self.adj_list.values()) / n) if n else 0,
            'critical_facilities': len(self.critical_facilities),
            'neighborhoods': len(self.neighborhoods)
        }

    def validate_neighborhood_coverage(self):
        # Ensure every neighborhood is connected to at least one critical facility
        return all(
            any(neigh == c or neigh in self.adj_list[c] for c in self.critical_facilities)
            for neigh in self.neighborhoods
        )

    def analyze_critical_paths(self):
        # Return list of edges connected to any critical facility
        return [
            (s, d) for s, d, _, _ in self.edges
            if s in self.critical_facilities or d in self.critical_facilities
        ]

    def dijkstra_traffic(self, start, end, time_period):
        # Modified Dijkstra's algorithm accounting for traffic at given time

        # Map time period to index (used in traffic flow array)
        time_index = {
            "morning": 0,
            "afternoon": 1,
            "evening": 2,
            "night": 3
        }[time_period]

        # Parse traffic flow string keys into tuple keys
        traffic_flow_lookup = {
            tuple(map(int, key.split('⟶') if '⟶' in key else key.split('-'))): values
            for key, values in self.traffic_flow.items()
        }

        # Initialize distances and priority queue
        distances = {n: float('inf') for n in self.nodes}
        prev = {n: None for n in self.nodes}
        distances[start] = 0
        pq = [(0, start)]

        while pq:
            curr_d, u = heapq.heappop(pq)
            if u == end:
                break
            if curr_d > distances[u]:
                continue

            for v, cond in self.adj_list[u]:
                # Get full edge details
                edge = next(
                    ((s, t, d, c) for s, t, d, c in self.edges if (s == u and t == v) or (s == v and t == u)),
                    None
                )
                if edge is None:
                    continue

                s, t, dist, capacity = edge

                # Get volume of traffic
                flow_key = (s, t) if (s, t) in traffic_flow_lookup else (t, s)
                flow_data = traffic_flow_lookup.get(flow_key, [capacity] * 4)
                volume = flow_data[time_index]

                # Apply congestion penalty
                congestion = volume / capacity if capacity > 0 else 1.0
                congestion_penalty = 1 + (congestion - 1) * 2  # tunable factor
                cost = (dist / cond if cond > 0 else float('inf')) * congestion_penalty

                # Relax edge
                nd = curr_d + cost
                if nd < distances[v]:
                    distances[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        # Reconstruct path
        if distances[end] == float('inf'):
            return [], float('inf')

        path = []
        node = end
        while node:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, distances[end]

    def euclidean_distance(self, n1, n2):
        # Compute Euclidean distance between two nodes
        x1, y1 = self.node_metadata[n1]['x'], self.node_metadata[n1]['y']
        x2, y2 = self.node_metadata[n2]['x'], self.node_metadata[n2]['y']
        return math.hypot(x2 - x1, y2 - y1)

    def a_star(self, start, goal):
        # A* Search Algorithm for finding optimal path

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}

        g_score = {n: float('inf') for n in self.nodes}
        f_score = {n: float('inf') for n in self.nodes}
        g_score[start] = 0
        f_score[start] = self.euclidean_distance(start, goal)

        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct path from goal to start
                path = []
                node = goal
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path, g_score[goal]

            closed.add(current)
            for neighbor, cond in self.adj_list[current]:
                # Get distance of edge
                dist = next(
                    (d for s, t, d, c in self.edges
                     if (s == current and t == neighbor) or (s == neighbor and t == current)),
                    float('inf')
                )
                tentative_g = g_score[current] + (dist / cond if cond != 0 else float('inf'))

                if neighbor in closed and tentative_g >= g_score[neighbor]:
                    continue

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.euclidean_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [], float('inf')
