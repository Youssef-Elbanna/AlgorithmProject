class RoadNetworkVisualizer:
    def __init__(self, road_network):
        # Ensure that road_network is correctly passed and set
        if not hasattr(road_network, 'nodes'):
            raise ValueError("Invalid road network object.")
        self.road_network = road_network

    def visualize_custom_edges(self, edges, ax=None, title=""):
        import networkx as nx

        G = nx.Graph()

        # Add all nodes to the graph
        for node_id, node_type in self.road_network.nodes.items():
            G.add_node(node_id, label=node_type)

        # Add only the selected edges
        for edge in edges:
            if len(edge) == 3:  # Handle edge format (node1, node2, cost)
                node1, node2, cost = edge
                G.add_edge(node1, node2, weight=cost)
            elif len(edge) == 4:  # Handle edge format (node1, node2, cost, maintenance)
                node1, node2, cost, _ = edge
                G.add_edge(node1, node2, weight=cost)
            else:
                raise ValueError(f"Unexpected edge format: {edge}")

        pos = nx.spring_layout(G)  # or use your own positions if you have them
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}, ax=ax)

        ax.set_title(title)
