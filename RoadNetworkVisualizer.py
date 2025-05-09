import networkx as nx
import matplotlib.pyplot as plt


class RoadNetworkVisualizer:
    def __init__(self, road_network):
        if not hasattr(road_network, 'nodes'):
            raise ValueError("Invalid road network object.")
        self.road_network = road_network

    def visualize_custom_edges(self, edges, ax=None, title="", layout_k=150.0, use_real_coords=False):
        G = nx.Graph()

        # Only include nodes used in edges
        node_ids_in_edges = set()
        for u, v, w in edges:
            node_ids_in_edges.update([u, v])
            G.add_edge(u, v, weight=w)

        for nid in node_ids_in_edges:
            ntype = self.road_network.nodes.get(nid, "N/A")
            G.add_node(nid, label=ntype)

        # Default spring layout
        pos = nx.spring_layout(G, k=layout_k, seed=42, scale=400.0)

        if use_real_coords:
            # Overwrite with normalized real positions if available
            real_pos = {
                nid: (meta['x'], meta['y'])
                for nid, meta in self.road_network.node_metadata.items()
                if nid in node_ids_in_edges and meta.get('x') is not None and meta.get('y') is not None
            }
            if real_pos:
                xs, ys = zip(*real_pos.values())
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                width = xmax - xmin or 1
                height = ymax - ymin or 1
                SCALE = 400.0
                for nid, (x, y) in real_pos.items():
                    pos[nid] = ((x - xmin) / width * SCALE, (y - ymin) / height * SCALE)

        # Draw network
        nx.draw(
            G, pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            node_size=700,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # Draw edge weights
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

        if ax:
            ax.set_title(title)

    def visualize_dijkstra_path(self, start, end, ax=None):
        path, cost = self.road_network.dijkstra(start, end)
        edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for s, t, d, c in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    edges.append((u, v, d))  # Use actual distance
                    break
        self.visualize_custom_edges(edges, ax=ax, title=f"Dijkstra {start}→{end} (Cost: {cost:.2f})")

    def visualize_a_star_path(self, start, end, ax=None):
        path, cost = self.road_network.a_star(start, end)
        edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for s, t, d, c in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    edges.append((u, v, d))  # Use actual distance
                    break
        self.visualize_custom_edges(edges, ax=ax, title=f"A* {start}→{end} (Cost: {cost:.2f})")
