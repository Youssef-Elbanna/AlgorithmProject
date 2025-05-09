import networkx as nx
import matplotlib.pyplot as plt


class RoadNetworkVisualizer:
    def __init__(self, road_network):
        if not hasattr(road_network, 'nodes'):
            raise ValueError("Invalid road network object.")
        self.road_network = road_network

    def visualize_custom_edges(self, edges, ax=None, title=""):
        G = nx.Graph()
        pos = {}

        # Add nodes with real (x, y) positions if available
        for nid, ntype in self.road_network.nodes.items():
            G.add_node(nid, label=ntype)
            meta = self.road_network.node_metadata.get(nid, {})
            if meta.get('x') is not None and meta.get('y') is not None:
                pos[nid] = (meta['x'], meta['y'])

        # Add edges
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        # --- Normalize coordinates and scale ---
        if pos:
            xs = [xy[0] for xy in pos.values()]
            ys = [xy[1] for xy in pos.values()]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            width, height = xmax - xmin, ymax - ymin

            # Avoid division by zero
            width = width or 1
            height = height or 1

            SCALE = 300.0  # Spread the graph more
            for nid, (x, y) in pos.items():
                nx_norm = (x - xmin) / width
                ny_norm = (y - ymin) / height
                pos[nid] = (nx_norm * SCALE, ny_norm * SCALE)

        # Add positions for missing nodes
        missing = [n for n in G.nodes if n not in pos]
        if missing:
            spring_pos = nx.spring_layout(G, k=50.0, seed=42)
            for n in missing:
                pos[n] = spring_pos[n]

        # Draw the network
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

        # Draw edge weights as labels
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
