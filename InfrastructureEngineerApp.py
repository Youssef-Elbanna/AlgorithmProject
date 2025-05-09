import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import networkx as nx
from collections import defaultdict
from RoadNetwork import RoadNetwork
from RoadNetworkVisualizer import RoadNetworkVisualizer

class InfrastructureEngineerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Infrastructure Engineer GUI")
        self.root.geometry("1000x700")

        self.road_network = RoadNetwork()
        self.visualizer = RoadNetworkVisualizer(self.road_network)
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Button(frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Generate MST", command=self.generate_mst).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Show Report", command=self.show_report).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Run Dijkstra", command=self.run_dijkstra).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Run A* Algorithm", command=self.run_a_star).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Traffic Flow Dijkstra", command=self.run_traffic_flow_dijkstra).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Greedy Algo", command=self.run_greedy_algo).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        try:
            nodes_fp = filedialog.askopenfilename(title="Select nodes CSV", filetypes=[("CSV","*.csv")])
            roads_fp = filedialog.askopenfilename(title="Select roads CSV", filetypes=[("CSV","*.csv")])
            nodes_df = pd.read_csv(nodes_fp)
            roads_df = pd.read_csv(roads_fp)

            self.road_network = RoadNetwork()
            self.visualizer = RoadNetworkVisualizer(self.road_network)

            for _, r in nodes_df.iterrows():
                self.road_network.add_node(
                    str(r["ID"]), r.get("Type","regular"),
                    float(r.get("X",0)), float(r.get("Y",0))
                )
            for _, r in roads_df.iterrows():
                self.road_network.add_edge(
                    str(r["From"]), str(r["To"]),
                    float(r["Distance"]), float(r["Condition"])
                )

            messagebox.showinfo("Data Loaded", "Successfully loaded nodes & roads.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def generate_mst(self):
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network loaded.")
            return
        edges, cost = self.road_network.kruskal_mst()
        self.display_graph(edges, f"MST (Cost: {cost:.2f})")

    def show_report(self):
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network loaded.")
            return
        c = self.road_network.calculate_total_cost()
        m = self.road_network.calculate_connectivity_metrics()
        ok = self.road_network.validate_neighborhood_coverage()
        report = (
            f"Construction: {c['construction_cost']:.2f}\n"
            f"Maintenance:  {c['maintenance_cost']:.2f}\n"
            f"First-Year:   {c['first_year_total']:.2f}\n\n"
            f"Nodes:   {m['total_nodes']}\n"
            f"Edges:   {m['total_edges']}\n"
            f"Avg deg: {m['avg_degree']:.2f}\n"
            f"Critical:{m['critical_facilities']}\n"
            f"Neighborhoods:{m['neighborhoods']}\n\n"
            f"Coverage: {'Adequate' if ok else 'Insufficient'}"
        )
        messagebox.showinfo("Network Report", report)

    def run_dijkstra(self):
        start = simpledialog.askstring("Dijkstra", "Start node ID:")
        end   = simpledialog.askstring("Dijkstra", "End node ID:")
        if not start or not end or start not in self.road_network.nodes or end not in self.road_network.nodes:
            messagebox.showerror("Error", "Invalid start/end.")
            return

        path, total_cost = self.road_network.dijkstra(start, end)
        if not path:
            messagebox.showerror("Error", "No path found.")
            return

        path_edges = []
        for u, v in zip(path, path[1:]):
            for s, t, dist, cond in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    cost_uv = dist / cond if cond else float('inf')
                    path_edges.append((u, v, cost_uv))
                    break

        self.display_graph(path_edges, f"Dijkstra {start}→{end} (Cost: {total_cost:.2f})")
        messagebox.showinfo("Dijkstra Result", f"Path: {' → '.join(path)}\nTotal Cost: {total_cost:.2f}")

    def run_a_star(self):
        start = simpledialog.askstring("A*", "Start node ID:")
        end   = simpledialog.askstring("A*", "End node ID:")
        if not start or not end or start not in self.road_network.nodes or end not in self.road_network.nodes:
            messagebox.showerror("Error", "Invalid start/end.")
            return

        path, total_cost = self.road_network.a_star(start, end)
        if not path:
            messagebox.showerror("Error", "No path found.")
            return

        path_edges = []
        for u, v in zip(path, path[1:]):
            for s, t, dist, cond in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    cost_uv = dist / cond if cond else float('inf')
                    path_edges.append((u, v, cost_uv))
                    break

        self.display_graph(path_edges, f"A* {start}→{end} (Cost: {total_cost:.2f})")
        messagebox.showinfo("A* Result", f"Path: {' → '.join(path)}\nTotal Cost: {total_cost:.2f}")

    def run_traffic_flow_dijkstra(self):
        def add_road(graph, node1, node2, traffic_volume):
            graph.add_edge(node1, node2, weight=traffic_volume)

        traffic_data = [
            ('1', '3', 2800, 1500, 2600, 800),
            ('1', '8', 2200, 1200, 2100, 600),
            ('2', '3', 2700, 1400, 2500, 700),
            ('2', '5', 3000, 1600, 2800, 650),
            ('3', '5', 3200, 1700, 3100, 800),
            ('3', '6', 1800, 1400, 1900, 500),
            ('3', '9', 2400, 1300, 2200, 550),
            ('3', '10', 2300, 1200, 2100, 500),
            ('4', '2', 3600, 1800, 3300, 750),
            ('4', '14', 2800, 1600, 2600, 600),
            ('5', '11', 2900, 1500, 2700, 650),
            ('6', '9', 1700, 1300, 1800, 450),
            ('7', '8', 3200, 1700, 3000, 700),
            ('7', '15', 2800, 1500, 2600, 600),
            ('8', '10', 2000, 1100, 1900, 450),
            ('8', '12', 2400, 1300, 2200, 500),
            ('9', '10', 1800, 1200, 1700, 400),
            ('10', '11', 2200, 1300, 2100, 500),
            ('11', 'F2', 2100, 1200, 2000, 450),
            ('12', '1', 2600, 1400, 2400, 550),
            ('13', '4', 3800, 2000, 3500, 800),
            ('14', '13', 3600, 1900, 3300, 750),
            ('15', '7', 2800, 1500, 2600, 600),
            ('F1', '5', 3300, 2200, 3100, 1200),
            ('F1', '2', 3000, 2000, 2800, 1100),
            ('F2', '3', 1900, 1600, 1800, 900),
            ('F7', '15', 2600, 1500, 2400, 550),
            ('F8', '4', 2800, 1600, 2600, 600)
        ]

        def get_time_period(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'

        def remove_heavy_traffic_edges(graph, threshold=2700):
            to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] > threshold]
            graph.remove_edges_from(to_remove)

        def on_submit():
            source = entry_from.get().strip()
            target = entry_to.get().strip()
            try:
                hour = int(entry_hour.get().strip())
                if not (0 <= hour <= 23):
                    raise ValueError
            except:
                messagebox.showerror("Error", "Hour must be between 0 and 23")
                return

            period = get_time_period(hour)
            G_original = nx.Graph()
            for u, v, m, a, e, n in traffic_data:
                val = {'morning': m, 'afternoon': a, 'evening': e, 'night': n}[period]
                add_road(G_original, u, v, val)

            try:
                path_before = nx.dijkstra_path(G_original, source, target, weight='weight')
                cost_before = nx.dijkstra_path_length(G_original, source, target, weight='weight')
                before_text = f"[Before Removal]\nPath: {path_before}\nCost: {cost_before:.2f}\n"
            except:
                before_text = "[Before Removal] No path found.\n"

            G_modified = G_original.copy()
            remove_heavy_traffic_edges(G_modified)

            try:
                path_after = nx.dijkstra_path(G_modified, source, target, weight='weight')
                cost_after = nx.dijkstra_path_length(G_modified, source, target, weight='weight')
                after_text = f"[After Removal]\nPath: {path_after}\nCost: {cost_after:.2f}"
            except:
                after_text = "[After Removal] No path found."

            output_label.config(text=before_text + "\n" + after_text)

        win = tk.Toplevel(self.root)
        win.title("Traffic Flow Dijkstra")
        win.geometry("500x400")

        tk.Label(win, text="FROM:").pack(pady=5)
        entry_from = tk.Entry(win)
        entry_from.pack(pady=5)

        tk.Label(win, text="TO:").pack(pady=5)
        entry_to = tk.Entry(win)
        entry_to.pack(pady=5)

        tk.Label(win, text="HOUR (0–23):").pack(pady=5)
        entry_hour = tk.Entry(win)
        entry_hour.pack(pady=5)

        tk.Button(win, text="Find Route", command=on_submit).pack(pady=15)
        output_label = tk.Label(win, text="", wraplength=450, justify="left", fg="blue")
        output_label.pack(pady=20)

    def run_greedy_algo(self):
        def compute_priority(road):
            return road["capacity"] / road["distance"]

        road_data = [
            {"from": 1, "to": 3, "distance": 8.5, "capacity": 3000},
            {"from": 1, "to": 8, "distance": 6.2, "capacity": 2500},
            {"from": 2, "to": 3, "distance": 5.9, "capacity": 2800},
            {"from": 2, "to": 5, "distance": 4.0, "capacity": 3200},
            {"from": 3, "to": 5, "distance": 6.1, "capacity": 3500},
            {"from": 3, "to": 6, "distance": 3.2, "capacity": 2000},
            {"from": 3, "to": 9, "distance": 4.5, "capacity": 2600},
            {"from": 3, "to": 10, "distance": 3.8, "capacity": 2400},
            {"from": 8, "to": 10, "distance": 3.3, "capacity": 2200},
            {"from": 9, "to": 10, "distance": 2.1, "capacity": 1900},
            {"from": 10, "to": 11, "distance": 8.7, "capacity": 2400},
            {"from": 11, "to": 2, "distance": 3.6, "capacity": 2200}
        ]

        graph = defaultdict(list)
        for r in road_data:
            graph[r["from"]].append(r)

        def greedy_path(current, end, visited):
            if current == end:
                return []
            visited.add(current)
            options = [r for r in graph[current] if r["to"] not in visited]
            if not options:
                return None
            options.sort(key=compute_priority, reverse=True)
            for best in options:
                sub_path = greedy_path(best["to"], end, visited.copy())
                if sub_path is not None:
                    return [best] + sub_path
            return None

        def on_submit():
            try:
                start = int(entry_from.get().strip())
                end = int(entry_to.get().strip())
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid integers for nodes.")
                return
            route = greedy_path(start, end, set())
            if route:
                output = f"Greedy Route from {start} to {end}:\n"
                total_distance = 0
                for step in route:
                    output += f"{step['from']} → {step['to']} | Distance: {step['distance']} km, Capacity: {step['capacity']}\n"
                    total_distance += step["distance"]
                output += f"\nTotal Distance: {round(total_distance, 2)} km"
            else:
                output = "No valid path found."
            output_label.config(text=output)

        greedy_root = tk.Toplevel(self.root)
        greedy_root.title("Greedy Route Finder")
        greedy_root.geometry("500x400")

        tk.Label(greedy_root, text="FROM (Node ID):").pack(pady=5)
        entry_from = tk.Entry(greedy_root)
        entry_from.pack(pady=5)

        tk.Label(greedy_root, text="TO (Node ID):").pack(pady=5)
        entry_to = tk.Entry(greedy_root)
        entry_to.pack(pady=5)

        tk.Button(greedy_root, text="Find Greedy Route", command=on_submit).pack(pady=15)

        output_label = tk.Label(greedy_root, text="", wraplength=450, justify="left", fg="darkgreen")
        output_label.pack(pady=20)


    def clear_canvas(self):
        for w in self.canvas_frame.winfo_children():
            w.destroy()

    def display_graph(self, edge_subset, title="Graph"):
        self.clear_canvas()
        fig = plt.Figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        self.visualizer.visualize_custom_edges(edge_subset, ax=ax, title=title)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = InfrastructureEngineerApp(root)
    root.mainloop()
