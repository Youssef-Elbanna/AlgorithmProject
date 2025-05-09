# RoadNetwork.py
from collections import defaultdict
import heapq
import math
from UnionFind import UnionFind

class RoadNetwork:
    def __init__(self):
        self.nodes = {}             # node_id -> type
        self.node_metadata = {}     # node_id -> {type, x, y}
        self.edges = []             # list of (src, dest, distance, condition)
        self.adj_list = defaultdict(list)  # node_id -> list of (neighbor, condition)
        self.critical_facilities = set()
        self.neighborhoods = set()

    def add_node(self, node_id, node_type='regular', x=None, y=None):
        self.nodes[node_id] = node_type
        self.node_metadata[node_id] = {'type': node_type, 'x': x, 'y': y}
        t = node_type.lower()
        if t in ['critical', 'airport', 'education', 'medical', 'government', 'transit hub']:
            self.critical_facilities.add(node_id)
        elif t in ['neighborhood', 'residential', 'mixed']:
            self.neighborhoods.add(node_id)

    def add_edge(self, src, dest, distance, condition):
        self.edges.append((src, dest, distance, condition))
        self.adj_list[src].append((dest, condition))
        self.adj_list[dest].append((src, condition))

    def _adjust_weight_for_critical(self, src, dest, weight):
        if src in self.critical_facilities or dest in self.critical_facilities:
            return weight * 0.8
        return weight

    def kruskal_mst(self):
        uf = UnionFind(self.nodes)
        # Use distance as the cost for MST, with critical bias
        weighted_edges = [
            (s, d, self._adjust_weight_for_critical(s, d, dist))
            for s, d, dist, _ in self.edges
        ]
        weighted_edges.sort(key=lambda x: x[2])

        mst = []
        total_cost = 0
        for src, dest, w in weighted_edges:
            if uf.union(src, dest):
                mst.append((src, dest, w))
                total_cost += w
        return mst, total_cost

    def calculate_total_cost(self):
        construction = sum(dist for _, _, dist, _ in self.edges)
        maintenance = construction * 0.1
        return {
            'construction_cost': construction,
            'maintenance_cost': maintenance,
            'first_year_total': construction + maintenance
        }

    def calculate_connectivity_metrics(self):
        n = len(self.nodes)
        return {
            'total_nodes': n,
            'total_edges': len(self.edges),
            'avg_degree': (sum(len(v) for v in self.adj_list.values()) / n) if n else 0,
            'critical_facilities': len(self.critical_facilities),
            'neighborhoods': len(self.neighborhoods)
        }

    def validate_neighborhood_coverage(self):
        return all(
            any(neigh == c or neigh in self.adj_list[c] for c in self.critical_facilities)
            for neigh in self.neighborhoods
        )

    def analyze_critical_paths(self):
        return [
            (s, d) for s, d, _, _ in self.edges
            if s in self.critical_facilities or d in self.critical_facilities
        ]

    def dijkstra(self, start, end):
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
                dist = next(
                    (d for s, t, d, c in self.edges
                     if (s == u and t == v) or (s == v and t == u)),
                    float('inf')
                )
                cost = dist / cond if cond != 0 else float('inf')
                nd = curr_d + cost
                if nd < distances[v]:
                    distances[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

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
        x1, y1 = self.node_metadata[n1]['x'], self.node_metadata[n1]['y']
        x2, y2 = self.node_metadata[n2]['x'], self.node_metadata[n2]['y']
        return math.hypot(x2 - x1, y2 - y1)

    def a_star(self, start, goal):
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
