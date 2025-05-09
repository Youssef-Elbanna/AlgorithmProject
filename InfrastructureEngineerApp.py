import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from RoadNetwork import RoadNetwork
from RoadNetworkVisualizer import RoadNetworkVisualizer
import pandas as pd


class InfrastructureEngineerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Infrastructure Engineer GUI")
        self.root.geometry("1000x700")

        self.road_network = RoadNetwork()
        self.mst_edges = []

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        load_btn = tk.Button(frame, text="Load Data", command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=10)

        mst_btn = tk.Button(frame, text="Generate MST", command=self.generate_mst)
        mst_btn.pack(side=tk.LEFT, padx=10)

        report_btn = tk.Button(frame, text="Show Report", command=self.show_report)
        report_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = tk.Button(frame, text="Clear Canvas", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        try:
            # You can replace these with filedialog if needed
            nodes_df = pd.read_csv("nodes.csv")
            roads_df = pd.read_csv("roads.csv")

            self.road_network = RoadNetwork()

            # Load nodes into the network
            for _, row in nodes_df.iterrows():
                self.road_network.add_node(str(row["ID"]), row.get("Type", "NODE"))

            # Load edges into the network
            for _, row in roads_df.iterrows():
                # Use the "Distance" as the cost for the edge
                # If you have maintenance cost, you can add it as an additional column
                self.road_network.add_edge(
                    str(row["From"]),
                    str(row["To"]),
                    float(row["Distance"]),
                    0  # Optional: you can add maintenance cost if your class supports it
                )

            messagebox.showinfo("Data Loaded", "Successfully loaded nodes.csv and roads.csv.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def generate_mst(self):
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network data loaded.")
            return

        self.mst_edges, total_cost = self.road_network.kruskal_mst()
        self.display_graph(self.mst_edges, title=f"Minimum Spanning Tree (Cost: {total_cost:.2f})")

    def show_report(self):
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network data loaded.")
            return

        costs = self.road_network.calculate_total_cost()
        metrics = self.road_network.calculate_connectivity_metrics()
        coverage_ok = self.road_network.validate_neighborhood_coverage()

        report = (
            f"Construction Cost: {costs['construction_cost']:.2f}\n"
            f"Maintenance Cost: {costs['maintenance_cost']:.2f}\n"
            f"First Year Total Cost: {costs['first_year_total']:.2f}\n\n"
            f"Total Nodes: {metrics['total_nodes']}\n"
            f"Total Edges: {metrics['total_edges']}\n"
            f"Average Degree: {metrics['avg_degree']:.2f}\n"
            f"Critical Facilities: {metrics['critical_facilities']}\n"
            f"Neighborhoods: {metrics['neighborhoods']}\n\n"
            f"Neighborhood Coverage: {'Adequate' if coverage_ok else 'Insufficient'}"
        )

        messagebox.showinfo("Network Report", report)

    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def display_graph(self, edge_subset, title="Graph"):
        self.clear_canvas()
        fig = plt.Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        visualizer = RoadNetworkVisualizer(self.road_network)
        visualizer.visualize_custom_edges(edge_subset, ax=ax, title=title)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = InfrastructureEngineerApp(root)
    root.mainloop()
