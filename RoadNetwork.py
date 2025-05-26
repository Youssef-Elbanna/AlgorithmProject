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
        # Generate Minimum Spanning Tree using Kruskal's algorithm
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

    def load_traffic_data(self, traffic_file):
        """Load traffic data from CSV file with time periods."""
        import csv
        self.traffic_flow = {}
        try:
            with open(traffic_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    road_id = row['RoadID']
                    # Store traffic data as [morning, afternoon, evening, night]
                    self.traffic_flow[road_id] = [
                        int(row['Morning']),
                        int(row['Afternoon']),
                        int(row['Evening']),
                        int(row['Night'])
                    ]
            print(f"Loaded traffic data for {len(self.traffic_flow)} roads")
        except Exception as e:
            print(f"Error loading traffic data: {e}")

    def dijkstra_traffic(self, start, end, time_period):
        """Dijkstra's algorithm with robust traffic and time period consideration."""
        print(f"\n=== Dijkstra's Algorithm ===")
        print(f"Finding path from {start} to {end} during {time_period}")
        
        # Map time_period to index, default to 'morning' if invalid
        time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
        time_index = time_map.get(str(time_period).lower(), 0)
        print(f"Using time index: {time_index} for {time_period}")
        
        distances = {n: float('inf') for n in self.nodes}
        prev = {n: None for n in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        print("\nStarting path search...")
        while pq:
            curr_d, u = heapq.heappop(pq)
            if u == end:
                print(f"\nReached destination node {end}")
                break
            if u in visited:
                continue
            visited.add(u)
            print(f"\nExploring node {u} (current distance: {curr_d:.2f})")
            
            for v, cond in self.adj_list.get(u, []):
                edge = next(((s, t, d, c) for s, t, d, c in self.edges if (s == u and t == v) or (s == v and t == u)), None)
                if edge is None:
                    print(f"  No edge found between {u} and {v}")
                    continue
                    
                s, t, dist, capacity = edge
                # Get traffic data for this edge
                edge_key = f"{s}-{t}" if f"{s}-{t}" in self.traffic_flow else f"{t}-{s}"
                flow_data = self.traffic_flow.get(edge_key, [capacity]*4)
                volume = flow_data[time_index]
                
                # Calculate congestion and cost
                congestion = volume / capacity if capacity > 0 else 1.0
                # Normalize congestion to be between 0 and 1
                congestion = min(congestion, 2.0) / 2.0
                # Base cost is normalized distance (assuming max distance is 10)
                base_cost = min(dist / 10.0, 1.0)
                # Condition factor (1.0 for perfect condition, 2.0 for worst condition)
                condition_factor = 1.0 + (1.0 - min(cond / 10.0, 1.0))
                # Traffic factor (1.0 for no traffic, 2.0 for max traffic)
                traffic_factor = 1.0 + congestion
                # Final cost is normalized to be between 0 and 99
                cost = (base_cost * condition_factor * traffic_factor) * 25
                
                print(f"  Edge {u}->{v}:")
                print(f"    Distance: {dist:.2f}")
                print(f"    Condition: {cond:.2f}")
                print(f"    Capacity: {capacity}")
                print(f"    Traffic volume ({time_period}): {volume}")
                print(f"    Congestion: {congestion:.2f}")
                print(f"    Base cost: {base_cost:.2f}")
                print(f"    Condition factor: {condition_factor:.2f}")
                print(f"    Traffic factor: {traffic_factor:.2f}")
                print(f"    Final cost: {cost:.2f}")
                
                nd = curr_d + cost
                if nd < distances[v]:
                    distances[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
                    print(f"    Updated distance to {v}: {nd:.2f}")
                else:
                    print(f"    No improvement for {v} (current: {distances[v]:.2f}, new: {nd:.2f})")

        if distances[end] == float('inf'):
            print(f"\nNo path found from {start} to {end}")
            return [], float('inf')
            
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        
        print(f"\n=== Path Found ===")
        print(f"Path: {' -> '.join(map(str, path))}")
        print(f"Total cost: {distances[end]:.2f}")
        print("==================\n")
        return path, distances[end]

    def euclidean_distance(self, n1, n2):
        # Compute Euclidean distance between two nodes
        x1, y1 = self.node_metadata[n1]['x'], self.node_metadata[n1]['y']
        x2, y2 = self.node_metadata[n2]['x'], self.node_metadata[n2]['y']
        return math.hypot(x2 - x1, y2 - y1)

    def a_star(self, start, goal, time_period="morning"):
        """A* algorithm with robust traffic and time period consideration."""
        print(f"\n=== A* Algorithm ===")
        print(f"Finding path from {start} to {goal} during {time_period}")
        
        # Map time_period to index, default to 'morning' if invalid
        time_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
        time_index = time_map.get(str(time_period).lower(), 0)
        print(f"Using time index: {time_index} for {time_period}")
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {n: float('inf') for n in self.nodes}
        f_score = {n: float('inf') for n in self.nodes}
        g_score[start] = 0
        f_score[start] = self.euclidean_distance(start, goal)
        closed = set()
        
        print("\nStarting path search...")
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                print(f"\nReached destination node {goal}")
                break
                
            closed.add(current)
            print(f"\nExploring node {current} (g_score: {g_score[current]:.2f}, f_score: {f_score[current]:.2f})")
            
            for neighbor, cond in self.adj_list.get(current, []):
                edge = next(((s, t, d, c) for s, t, d, c in self.edges if (s == current and t == neighbor) or (s == neighbor and t == current)), None)
                if edge is None:
                    print(f"  No edge found between {current} and {neighbor}")
                    continue
                    
                s, t, dist, capacity = edge
                # Get traffic data for this edge
                edge_key = f"{s}-{t}" if f"{s}-{t}" in self.traffic_flow else f"{t}-{s}"
                flow_data = self.traffic_flow.get(edge_key, [capacity]*4)
                volume = flow_data[time_index]
                
                # Calculate congestion and cost
                congestion = volume / capacity if capacity > 0 else 1.0
                # Normalize congestion to be between 0 and 1
                congestion = min(congestion, 2.0) / 2.0
                # Base cost is normalized distance (assuming max distance is 10)
                base_cost = min(dist / 10.0, 1.0)
                # Condition factor (1.0 for perfect condition, 2.0 for worst condition)
                condition_factor = 1.0 + (1.0 - min(cond / 10.0, 1.0))
                # Traffic factor (1.0 for no traffic, 2.0 for max traffic)
                traffic_factor = 1.0 + congestion
                # Final cost is normalized to be between 0 and 99
                cost = (base_cost * condition_factor * traffic_factor) * 25
                
                print(f"  Edge {current}->{neighbor}:")
                print(f"    Distance: {dist:.2f}")
                print(f"    Condition: {cond:.2f}")
                print(f"    Capacity: {capacity}")
                print(f"    Traffic volume ({time_period}): {volume}")
                print(f"    Congestion: {congestion:.2f}")
                print(f"    Base cost: {base_cost:.2f}")
                print(f"    Condition factor: {condition_factor:.2f}")
                print(f"    Traffic factor: {traffic_factor:.2f}")
                print(f"    Final cost: {cost:.2f}")
                
                tentative_g = g_score[current] + cost
                if neighbor in closed and tentative_g >= g_score[neighbor]:
                    print(f"    Skipping {neighbor} (already visited with better score)")
                    continue
                    
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.euclidean_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    print(f"    Updated scores for {neighbor}:")
                    print(f"    g_score: {g_score[neighbor]:.2f}")
                    print(f"    f_score: {f_score[neighbor]:.2f}")
                else:
                    print(f"    No improvement for {neighbor} (current g_score: {g_score[neighbor]:.2f}, new: {tentative_g:.2f})")

        if goal not in came_from:
            print(f"\nNo path found from {start} to {goal}")
            return [], float('inf')
            
        path = []
        node = goal
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        path.reverse()
        
        print(f"\n=== Path Found ===")
        print(f"Path: {' -> '.join(map(str, path))}")
        print(f"Total cost: {g_score[goal]:.2f}")
        print("==================\n")
        return path, g_score[goal]
