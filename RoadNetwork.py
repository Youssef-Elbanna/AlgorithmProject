from collections import defaultdict
from UnionFind import UnionFind

class RoadNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adj_list = defaultdict(list)
        self.critical_facilities = set()
        self.neighborhoods = set()

    def add_node(self, node_id, node_type='regular'):
        self.nodes[node_id] = node_type
        if node_type == 'critical':
            self.critical_facilities.add(node_id)
        elif node_type == 'neighborhood':
            self.neighborhoods.add(node_id)

    def add_edge(self, src, dest, distance, cost):
        self.edges.append((src, dest, cost))
        self.adj_list[src].append((dest, cost))
        self.adj_list[dest].append((src, cost))

    def _adjust_weight_for_critical(self, src, dest, weight):
        if src in self.critical_facilities or dest in self.critical_facilities:
            return weight * 0.8  # prioritize critical facility connections
        return weight

    def kruskal_mst(self):
        uf = UnionFind(self.nodes)
        mst = []
        total_cost = 0
        edges = [(s, d, self._adjust_weight_for_critical(s, d, w)) for s, d, w in self.edges]
        edges.sort(key=lambda x: x[2])

        for src, dest, weight in edges:
            if uf.union(src, dest):
                mst.append((src, dest, weight))
                total_cost += weight

        return mst, total_cost

    def calculate_total_cost(self):
        construction_cost = sum(edge[2] for edge in self.edges)
        maintenance_estimate = construction_cost * 0.1
        return {
            'construction_cost': construction_cost,
            'maintenance_cost': maintenance_estimate,
            'first_year_total': construction_cost + maintenance_estimate
        }

    def calculate_connectivity_metrics(self):
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'avg_degree': sum(len(neighbors) for neighbors in self.adj_list.values()) / len(self.nodes),
            'critical_facilities': len(self.critical_facilities),
            'neighborhoods': len(self.neighborhoods)
        }

    def validate_neighborhood_coverage(self):
        return all(any(neigh == node or neigh in self.adj_list[node] for node in self.critical_facilities) for neigh in self.neighborhoods)

    def analyze_critical_paths(self):
        return [(src, dest) for src, dest, _ in self.edges if src in self.critical_facilities or dest in self.critical_facilities]
