import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
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

        # Build the edge list with per-edge cost = distance/condition
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
