# Import necessary modules
import networkx as nx
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Text, Scrollbar, RIGHT, Y, BOTH, END
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from RoadNetwork import RoadNetwork
from RoadNetworkVisualizer import RoadNetworkVisualizer
from collections import defaultdict
from functools import lru_cache
import numpy as np
import os

# ---------- UI Theme Functions ----------

# Applies a modern theme to the root window and its children
def apply_theme(root):
    style = {
        "bg": "#fdfdfd",
        "fg": "#2c3e50",
        "accent": "#001F54",
        "button_hover": "#2e86de",
        "font": ("Segoe UI", 10),
        "title_font": ("Segoe UI", 14, "bold")
    }
    root.configure(bg=style["bg"])
    for widget in root.winfo_children():
        apply_widget_theme(widget, style)

# Recursively apply the theme to each widget
def apply_widget_theme(widget, style):
    if isinstance(widget, (tk.Frame, Toplevel)):
        widget.configure(bg=style["bg"])
    elif isinstance(widget, tk.Label):
        widget.configure(bg=style["bg"], fg=style["fg"], font=style["font"])
    elif isinstance(widget, tk.Button):
        widget.configure(bg=style["accent"], fg="white", font=style["font"],
                         activebackground=style["button_hover"],
                         activeforeground="white", relief=tk.FLAT, bd=1,
                         highlightthickness=0, padx=10, pady=5)
        widget.bind("<Enter>", lambda e: widget.config(bg=style["button_hover"]))
        widget.bind("<Leave>", lambda e: widget.config(bg=style["accent"]))
    elif isinstance(widget, (tk.Entry, tk.Text, tk.OptionMenu)):
        widget.configure(bg="#ffffff", fg=style["fg"], font=style["font"], relief=tk.SOLID, bd=1)
    for child in widget.winfo_children():
        apply_widget_theme(child, style)


# ---------- Transit Data ----------

# Format: route ID: (stop sequence, travel time, daily passengers)
bus_routes = {
    "B1": ("1,3,6,9", 25, 35000),
    "B2": ("7,15,8,10,3", 30, 42000),
    "B3": ("2,5,F1", 20, 28000),
    "B4": ("4,14,2,3", 22, 31000),
    "B5": ("8,12,1", 18, 25000),
    "B6": ("11,5,2", 24, 33000),
    "B7": ("13,4,14", 15, 21000),
    "B8": ("F7,15,7", 12, 17000),
    "B9": ("1,8,10,9,6", 28, 39000),
    "B10": ("F8,4,2,5", 20, 28000)
}

# Format: line ID: (stop sequence, daily passengers)
metro_lines = {
    "M1": ("12,1,3,F2,11", 1500000),
    "M2": ("11,F2,3,10,8", 1200000),
    "M3": ("F1,5,2,3,9", 800000)
}


# ---------- Optimization Logic ----------

# Dynamic programming to select bus routes under a fixed budget to maximize passengers
def optimize_bus_schedule(budget):
    bus_ids = list(bus_routes.keys())
    n = len(bus_ids)

    @lru_cache(None)
    def dp(i, remaining):
        if i == n:
            return 0, []
        skip_val, skip_list = dp(i + 1, remaining)
        bus_id = bus_ids[i]
        _, _, passengers = bus_routes[bus_id]
        if remaining >= 1:
            take_val, take_list = dp(i + 1, remaining - 1)
            take_val += passengers
            take_list = take_list + [bus_id]
            if take_val > skip_val:
                return take_val, take_list
        return skip_val, skip_list

    return dp(0, budget)


# Knapsack-style DP to prioritize road maintenance for most benefit within a km limit
def optimize_road_maintenance(roads_df, max_km=50):
    roads = []
    for _, row in roads_df.iterrows():
        if row['Condition'] <= 6:  # Only consider roads in poor condition
            benefit = row['Distance'] * (10 - row['Condition'])  # Heuristic for benefit
            roads.append((row['From'], row['To'], row['Distance'], benefit))
    n = len(roads)
    dp = np.zeros((n + 1, int(max_km * 10) + 1))
    for i in range(n - 1, -1, -1):
        for rem in range(int(max_km * 10) + 1):
            skip = dp[i + 1][rem]
            take = 0
            if rem >= int(roads[i][2] * 10):
                take = roads[i][3] + dp[i + 1][rem - int(roads[i][2] * 10)]
            dp[i][rem] = max(skip, take)

    # Backtrack to find selected roads
    selected = []
    rem = int(max_km * 10)
    for i in range(n):
        if dp[i][rem] != dp[i + 1][rem]:
            selected.append(roads[i])
            rem -= int(roads[i][2] * 10)
    return dp[0][int(max_km * 10)], selected


# Calculate metro line efficiency (passengers per unique stop)
def rank_metro_lines():
    scores = {}
    for line_id, (stops, passengers) in metro_lines.items():
        score = passengers / len(set(stops.split(",")))
        scores[line_id] = score
    return sorted(scores.items(), key=lambda x: -x[1])


# Find common stops between bus routes and metro lines
def find_transfer_points():
    transfer_points = defaultdict(list)
    metro_stops = defaultdict(set)
    for m_id, (m_stops, _) in metro_lines.items():
        for stop in m_stops.split(","):
            metro_stops[stop].add(m_id)
    for b_id, (b_stops, _, _) in bus_routes.items():
        for stop in b_stops.split(","):
            if stop in metro_stops:
                for m_id in metro_stops[stop]:
                    transfer_points[stop].append((b_id, m_id))
    return transfer_points


# Run all optimization analyses and return formatted result
def run_transit_optimization(roads_df):
    summary = []
    max_passengers, selected_buses = optimize_bus_schedule(30)
    summary.append("--- Optimizing Bus Schedule (DP) ---")
    summary.append(f"Selected Buses: {selected_buses}")
    summary.append(f"Max Daily Passengers: {max_passengers}\n")

    total_benefit, selected_roads = optimize_road_maintenance(roads_df)
    summary.append("--- Optimizing Road Maintenance (DP) ---")
    summary.append(f"Total Benefit: {total_benefit}")
    summary.append("Selected Roads:")
    for r in selected_roads:
        summary.append(f"{r[0]} → {r[1]} ({r[2]} km)")

    summary.append("\n--- Ranking Metro Lines by Coverage Efficiency ---")
    for line, score in rank_metro_lines():
        summary.append(f"{line}: Score = {score:.2f}")

    summary.append("\n--- Bus ↔ Metro Transfer Points ---")
    transfer_points = find_transfer_points()
    for stop, connections in transfer_points.items():
        for b_id, m_id in connections:
            summary.append(f"Stop {stop} connects Bus {b_id} with Metro {m_id}")
    return "\n".join(summary)


# ---------- Result Display Functions ----------

# Show all transit optimization results in a scrollable pop-up window
def show_transit_results(root, roads_df):
    result_text = run_transit_optimization(roads_df)
    window = Toplevel(root)
    window.title("Transit Optimization Report")
    window.geometry("800x600")
    text_area = Text(window, wrap='word')
    text_area.insert(END, result_text)
    text_area.config(state='disabled')
    scroll = Scrollbar(window, command=text_area.yview)
    text_area.config(yscrollcommand=scroll.set)
    text_area.pack(side='left', fill=BOTH, expand=True)
    scroll.pack(side=RIGHT, fill=Y)
    window.transient(root)
    window.grab_set()
    root.wait_window(window)


# Show an interactive dialog to explore transit analysis options
def show_transit_analysis_dialog(root, roads_df):
    def run_maintenance():
        try:
            km = float(km_entry.get())
            benefit, roads = optimize_road_maintenance(roads_df, max_km=km)
            result = f"Max Distance: {km} km\nTotal Benefit: {benefit}\n\nSelected Roads:\n"
            for r in roads:
                result += f"{r[0]} → {r[1]} ({r[2]} km)\n"
            output_text.delete(1.0, END)
            output_text.insert(END, result)
        except Exception as e:
            output_text.delete(1.0, END)
            output_text.insert(END, f"Error: {e}")

    def run_metro_efficiency():
        line = metro_var.get()
        if line in metro_lines:
            stops, passengers = metro_lines[line]
            score = passengers / len(set(stops.split(",")))
            output_text.delete(1.0, END)
            output_text.insert(END, f"Metro Line: {line}\nCoverage Efficiency: {score:.2f} passengers per stop")
        else:
            output_text.delete(1.0, END)
            output_text.insert(END, "Invalid metro line selected.")

    def run_transfer_check():
        stop = stop_entry.get()
        found = []
        transfers = find_transfer_points()
        if stop in transfers:
            for b, m in transfers[stop]:
                found.append(f"Bus {b} connects with Metro {m}")
            output_text.delete(1.0, END)
            output_text.insert(END, f"Stop {stop} Connections:\n" + "\n".join(found))
        else:
            output_text.delete(1.0, END)
            output_text.insert(END, f"No connections found for stop {stop}.")

    # Create dialog UI
    win = Toplevel(root)
    win.title("Transit Analysis Options")
    win.geometry("600x500")

    tk.Label(win, text="1. Road Maintenance - Enter Max Distance (km)").pack()
    km_entry = tk.Entry(win)
    km_entry.pack()
    tk.Button(win, text="Optimize Roads", command=run_maintenance).pack(pady=5)

    tk.Label(win, text="\n2. Metro Line Efficiency - Select Line").pack()
    metro_var = tk.StringVar(win)
    metro_var.set("M1")
    tk.OptionMenu(win, metro_var, *metro_lines.keys()).pack()
    tk.Button(win, text="Check Efficiency", command=run_metro_efficiency).pack(pady=5)

    tk.Label(win, text="\n3. Transfer Points - Enter Stop ID").pack()
    stop_entry = tk.Entry(win)
    stop_entry.pack()
    tk.Button(win, text="Check Transfer", command=run_transfer_check).pack(pady=5)

    tk.Label(win, text="\nResults:").pack()
    output_text = Text(win, height=10, wrap='word')
    output_text.pack(fill=BOTH, expand=True, padx=10, pady=5)

    win.transient(root)
    win.grab_set()
    root.wait_window(win)



# ---------- GUI Class ----------
class InfrastructureEngineerApp:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Infrastructure Engineer GUI")
        self.root.geometry("1000x700")
        
        # Initialize road network and its visualizer
        self.road_network = RoadNetwork()
        self.visualizer = RoadNetworkVisualizer(self.road_network)
        
        self.roads_df = None  # Placeholder for loaded road data as a DataFrame
        self.create_widgets()  # Create the GUI layout and buttons
        apply_theme(self.root)  # Apply consistent theme to the GUI
        
        # Automatically load data when program starts
        self.load_data()

    def create_widgets(self):
        # Header label
        header = tk.Label(self.root, text="🚧 Smart Infrastructure Management", font=("Segoe UI", 16, "bold"))
        header.pack(pady=15)

        # Frame for holding control buttons
        frame = tk.Frame(self.root, bg="#f0f4f8", relief=tk.RIDGE, bd=2, padx=10, pady=10)
        frame.pack(pady=10)

        # Button controls for various functionalities
        tk.Button(frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Generate MST", command=self.generate_mst).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Show Report", command=self.show_report).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Run Dijkstra algorithm", command=self.run_dijkstra).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Emergency Route finder", command=self.run_a_star).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Transit Optimization", command=lambda: show_transit_results(self.root, self.roads_df)).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Transit Analysis",command=lambda: show_transit_analysis_dialog(self.root, self.roads_df)).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Greedy Route Finder", command=self.show_greedy_route_finder).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Traffic Flow Dijkstra", command=self.show_traffic_dijkstra).pack(side=tk.LEFT, padx=5)

        # Canvas area for visual output
        self.canvas_frame = tk.Frame(self.root, bg="#ffffff")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_data(self):
        # Load node and edge data from CSV files automatically
        try:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Define the expected file paths
            nodes_fp = os.path.join(current_dir, "nodes.csv")
            roads_fp = os.path.join(current_dir, "roads.csv")
            traffic_fp = os.path.join(current_dir, "traffic.csv")
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [nodes_fp, roads_fp, traffic_fp]):
                messagebox.showerror("Error", "Required data files not found. Please ensure nodes.csv, roads.csv, and traffic.csv are in the same directory as the program.")
                return
            
            nodes_df = pd.read_csv(nodes_fp)
            roads_df = pd.read_csv(roads_fp)
            self.roads_df = roads_df

            # Re-initialize network and visualizer
            self.road_network = RoadNetwork()
            self.visualizer = RoadNetworkVisualizer(self.road_network)

            # Add nodes from CSV
            for _, r in nodes_df.iterrows():
                population = int(r["Population"]) if "Population" in r and not pd.isna(r["Population"]) else 0
                self.road_network.add_node(
                    str(r["ID"]),
                    r.get("Type", "regular"),
                    float(r.get("X", 0)),
                    float(r.get("Y", 0)),
                    population
                )

            # Add edges from CSV
            for _, r in roads_df.iterrows():
                self.road_network.add_edge(
                    str(r["From"]),
                    str(r["To"]),
                    float(r["Distance"]),
                    float(r["Condition"])
                )

            # Load traffic data
            self.road_network.load_traffic_data(traffic_fp)

            messagebox.showinfo("Data Loaded", "Successfully loaded nodes, roads, and traffic data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def generate_mst(self):
        # Generate and visualize a Minimum Spanning Tree using Kruskal's algorithm
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network loaded.")
            return
        edges, cost = self.road_network.kruskal_mst()
        self.display_graph(edges, f"MST (Cost: {cost:.2f})")

    def show_report(self):
        # Display summary statistics about the MST
        if not self.road_network.nodes:
            messagebox.showerror("Error", "No network loaded.")
            return

        mst_edges, cost = self.road_network.kruskal_mst()
        total_nodes = len(set([u for u, v, _ in mst_edges] + [v for u, v, _ in mst_edges]))
        total_edges = len(mst_edges)
        avg_degree = (2 * total_edges) / total_nodes if total_nodes > 0 else 0

        report = (
            f"--- MST Report ---\n"
            f"Total Cost:   {cost:.2f}\n\n"
            f"Nodes in MST: {total_nodes}\n"
            f"Edges in MST: {total_edges}\n"
            f"Avg Degree:   {avg_degree:.2f}"
        )
        messagebox.showinfo("MST Report", report)

    def run_dijkstra(self):
        # Prompt user for start, end, and time, and then run time-based Dijkstra algorithm
        try:
            start = simpledialog.askstring("Dijkstra", "Start node ID:")
            end = simpledialog.askstring("Dijkstra", "End node ID:")

            if not start or not end:
                messagebox.showerror("Error", "Start and End must be provided.")
                return

            start = start.strip()
            end = end.strip()
        except Exception:
            messagebox.showerror("Error", "Invalid input for start/end node.")
            return

        start_time_str = simpledialog.askstring("Dijkstra", "Start time (HH:MM, 24h format):")
        if not start_time_str:
            messagebox.showerror("Error", "Start time required.")
            return

        try:
            hour, minute = map(int, start_time_str.strip().split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
            start_time_minutes = hour * 60 + minute
        except ValueError:
            messagebox.showerror("Error", "Invalid time format. Use HH:MM.")
            return

        def get_time_period(minutes):
            if 360 <= minutes < 720:  # 06:00 - 12:00
                return "morning"
            elif 720 <= minutes < 1020:  # 12:00 - 17:00
                return "afternoon"
            elif 1020 <= minutes < 1320:  # 17:00 - 22:00
                return "evening"
            else:  # 22:00 - 06:00
                return "night"

        time_period = get_time_period(start_time_minutes)

        if start not in self.road_network.nodes or end not in self.road_network.nodes:
            messagebox.showerror("Error", "Invalid start/end node.")
            return

        path, total_cost = self.road_network.dijkstra_traffic(start, end, time_period)
        if not path:
            messagebox.showerror("Error", "No path found.")
            return

        # Build list of edges for visualizing path
        path_edges = []
        for u, v in zip(path, path[1:]):
            for s, t, dist, cond in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    cost_uv = dist / cond if cond else float('inf')
                    path_edges.append((u, v, cost_uv))
                    break

        self.display_graph(path_edges,
                           f"Dijkstra {start}→{end} at {start_time_str} ({time_period.title()}, Cost: {total_cost:.2f})")
        messagebox.showinfo("Dijkstra Result", f"Path: {' → '.join(path)}\nTotal Cost: {total_cost:.2f}")

    def run_a_star(self):
        # Prompt user for start, end, and time, and then run time-based A* algorithm
        try:
            start = simpledialog.askstring("A*", "Start node ID:")
            end = simpledialog.askstring("A*", "End node ID:")

            if not start or not end:
                messagebox.showerror("Error", "Start and End must be provided.")
                return

            start = start.strip()
            end = end.strip()
        except Exception:
            messagebox.showerror("Error", "Invalid input for start/end node.")
            return

        start_time_str = simpledialog.askstring("A*", "Start time (HH:MM, 24h format):")
        if not start_time_str:
            messagebox.showerror("Error", "Start time required.")
            return

        try:
            hour, minute = map(int, start_time_str.strip().split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
            start_time_minutes = hour * 60 + minute
        except ValueError:
            messagebox.showerror("Error", "Invalid time format. Use HH:MM.")
            return

        def get_time_period(minutes):
            if 360 <= minutes < 720:  # 06:00 - 12:00
                return "morning"
            elif 720 <= minutes < 1020:  # 12:00 - 17:00
                return "afternoon"
            elif 1020 <= minutes < 1320:  # 17:00 - 22:00
                return "evening"
            else:  # 22:00 - 06:00
                return "night"

        time_period = get_time_period(start_time_minutes)

        if start not in self.road_network.nodes or end not in self.road_network.nodes:
            messagebox.showerror("Error", "Invalid start/end node.")
            return

        path, total_cost = self.road_network.a_star(start, end, time_period)
        if not path:
            messagebox.showerror("Error", "No path found.")
            return

        # Build list of edges for visualizing path
        path_edges = []
        for u, v in zip(path, path[1:]):
            for s, t, dist, cond in self.road_network.edges:
                if (s == u and t == v) or (s == v and t == u):
                    cost_uv = dist / cond if cond else float('inf')
                    path_edges.append((u, v, cost_uv))
                    break

        self.display_graph(path_edges,
                           f"A* {start}→{end} at {start_time_str} ({time_period.title()}, Cost: {total_cost:.2f})")
        messagebox.showinfo("A* Result", f"Path: {' → '.join(path)}\nTotal Cost: {total_cost:.2f}")

    def clear_canvas(self):
        for w in self.canvas_frame.winfo_children():
            w.destroy()

    def display_graph(self, edge_subset, title="Graph"):
        self.clear_canvas()
        fig = plt.Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        self.visualizer.visualize_custom_edges(edge_subset, ax=ax, title=title, layout_k=2.5)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_greedy_route_finder(self):
        from_data = [
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
        G = defaultdict(list)
        for r in from_data:
            G[r["from"]].append(r)

        def compute_priority(road):
            return road["capacity"] / road["distance"]

        def greedy_path(current, end, visited):
            if current == end:
                return []
            visited.add(current)
            options = [r for r in G[current] if r["to"] not in visited]
            if not options:
                return None
            options.sort(key=compute_priority, reverse=True)
            for best in options:
                sub = greedy_path(best["to"], end, visited.copy())
                if sub is not None:
                    return [best] + sub
            return None

        win = Toplevel(self.root)
        win.title("Greedy Route Finder")
        win.geometry("500x400")
        tk.Label(win, text="From:").pack()
        e_from = tk.Entry(win)
        e_from.pack()
        tk.Label(win, text="To:").pack()
        e_to = tk.Entry(win)
        e_to.pack()
        result = tk.Label(win, text="", wraplength=480, justify="left")
        result.pack()

        def on_submit():
            try:
                start = int(e_from.get())
                end = int(e_to.get())
                path = greedy_path(start, end, set())
                if path:
                    text = "Route:\n"
                    total = 0
                    for r in path:
                        text += f"{r['from']} → {r['to']} | Distance: {r['distance']} km, Capacity: {r['capacity']}\n"
                        total += r["distance"]
                    text += f"Total Distance: {total:.2f} km"
                else:
                    text = "No route found."
                result.config(text=text)
            except:
                result.config(text="Error")

        tk.Button(win, text="Run", command=on_submit).pack(pady=5)

    def show_traffic_dijkstra(self):

        def get_time_period(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'

        def add_road(graph, node1, node2, traffic_volume):
            graph.add_edge(node1, node2, weight=traffic_volume)

        def remove_heavy_traffic_edges(graph, threshold=2700):
            to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] > threshold]
            graph.remove_edges_from(to_remove)

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

        win = Toplevel(self.root)
        win.title("Traffic Flow Dijkstra")
        win.geometry("500x500")

        tk.Label(win, text="FROM:").pack(pady=5)
        entry_from = tk.Entry(win)
        entry_from.pack(pady=5)

        tk.Label(win, text="TO:").pack(pady=5)
        entry_to = tk.Entry(win)
        entry_to.pack(pady=5)

        tk.Label(win, text="HOUR (0–23):").pack(pady=5)
        entry_hour = tk.Entry(win)
        entry_hour.pack(pady=5)

        output_label = tk.Label(win, text="", wraplength=450, justify="left", fg="blue")
        output_label.pack(pady=20)

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

            before_text = ""
            after_text = ""

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

        tk.Button(win, text="Find Route", command=on_submit).pack(pady=15)


if __name__ == "__main__":
    root = tk.Tk()
    app = InfrastructureEngineerApp(root)
    root.mainloop()