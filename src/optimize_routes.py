"""
Route Optimization Module

This module uses Google OR-Tools to solve TSP (Traveling Salesman Problem)
or VRP (Vehicle Routing Problem) for warehouse robots.

Features:
- Load distance matrix from data/raw/
- Solve TSP (1 robot) or VRP (N robots)
- Visualize optimized routes with colored paths and arrows
- Display metrics (total distance, solve time)
- Save results to data/processed/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
import json
from pathlib import Path
import time
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class RouteOptimizer:
    """
    Optimizes robot routes using OR-Tools for TSP or VRP problems.
    """
    
    def __init__(
        self,
        distance_matrix: np.ndarray,
        points: List[Tuple[float, float]],
        point_labels: List[str],
        n_robots: int = 1,
        depot_index: int = 0
    ):
        """
        Initialize the route optimizer.
        
        Args:
            distance_matrix: n×n numpy array of distances between points
            points: List of (x, y) coordinate tuples
            point_labels: List of point labels (e.g., ["robot_0", "pickup_0", ...])
            n_robots: Number of robots (1 = TSP, >1 = VRP)
            depot_index: Index of the depot/starting point (default: 0, first robot)
        """
        self.distance_matrix = distance_matrix.astype(int)  # OR-Tools requires integers
        self.points = points
        self.point_labels = point_labels
        self.n_robots = n_robots
        self.depot_index = depot_index
        self.n_points = len(points)
        
        # Solution storage
        self.solution = None
        self.routes = []
        self.route_distances = []
        self.total_distance = 0
        self.solve_time = 0
        
    def create_distance_callback(self, manager: pywrapcp.RoutingIndexManager) -> callable:
        """
        Create a distance callback function for OR-Tools.
        
        Args:
            manager: OR-Tools RoutingIndexManager
        
        Returns:
            Distance callback function
        """
        def distance_callback(from_index: int, to_index: int) -> int:
            """
            Returns the distance between two nodes.
            
            Args:
                from_index: Internal routing index of from node
                to_index: Internal routing index of to node
            
            Returns:
                Distance between the two nodes
            """
            # Convert from routing variable Index to distance matrix NodeIndex
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node])
        
        return distance_callback
    
    def solve(
        self,
        search_strategy: str = "PATH_CHEAPEST_ARC",
        time_limit_seconds: int = 30
    ) -> Dict:
        """
        Solve the TSP or VRP problem using OR-Tools.
        
        Args:
            search_strategy: Search strategy ("PATH_CHEAPEST_ARC" or "GUIDED_LOCAL_SEARCH")
            time_limit_seconds: Maximum time to spend solving (seconds)
        
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            self.n_points,
            self.n_robots,
            self.depot_index
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register distance callback
        distance_callback_index = routing.RegisterTransitCallback(
            self.create_distance_callback(manager)
        )
        
        # Set arc cost evaluator
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # Set search strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            if search_strategy == "PATH_CHEAPEST_ARC"
            else routing_enums_pb2.FirstSolutionStrategy.GUIDED_LOCAL_SEARCH
        )
        
        # Set local search metaheuristic
        if search_strategy == "GUIDED_LOCAL_SEARCH":
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = time_limit_seconds
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        self.solve_time = time.time() - start_time
        
        if solution:
            # Extract solution
            self.solution = solution
            self.routes = []
            self.route_distances = []
            self.total_distance = 0
            
            for vehicle_id in range(self.n_robots):
                route = []
                route_distance = 0
                index = routing.Start(vehicle_id)
                
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
                
                # Add depot at the end (return to start)
                route.append(manager.IndexToNode(index))
                self.routes.append(route)
                self.route_distances.append(route_distance)
                self.total_distance += route_distance
            
            return {
                "status": "OPTIMAL" if solution.ObjectiveValue() == self.total_distance else "FEASIBLE",
                "total_distance": self.total_distance,
                "solve_time": self.solve_time,
                "n_routes": len(self.routes),
                "route_distances": self.route_distances
            }
        else:
            return {
                "status": "NO_SOLUTION",
                "total_distance": None,
                "solve_time": self.solve_time,
                "n_routes": 0,
                "route_distances": []
            }
    
    def visualize(
        self,
        save_path: Optional[str] = None,
        show_arrows: bool = True
    ) -> None:
        """
        Visualize the optimized routes with colored paths and arrows.
        
        Args:
            save_path: Optional path to save the figure
            show_arrows: If True, show directional arrows on routes
        """
        if not self.solution:
            print("No solution to visualize. Run solve() first.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Color palette for different routes
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_robots))
        
        # Plot all points first
        all_x = [p[0] for p in self.points]
        all_y = [p[1] for p in self.points]
        
        # Plot robots (depot)
        robot_indices = [i for i, label in enumerate(self.point_labels) if label.startswith("robot")]
        if robot_indices:
            robot_x = [all_x[i] for i in robot_indices]
            robot_y = [all_y[i] for i in robot_indices]
            ax.scatter(robot_x, robot_y, c='blue', s=200, marker='s', 
                      label='Robots (Depot)', zorder=10, edgecolors='black', linewidths=2)
        
        # Plot pick-up points
        pickup_indices = [i for i, label in enumerate(self.point_labels) if label.startswith("pickup")]
        if pickup_indices:
            pickup_x = [all_x[i] for i in pickup_indices]
            pickup_y = [all_y[i] for i in pickup_indices]
            ax.scatter(pickup_x, pickup_y, c='green', s=120, marker='o', 
                      label='Pick-up Points', zorder=9, edgecolors='black', linewidths=1.5)
        
        # Plot delivery stations
        delivery_indices = [i for i, label in enumerate(self.point_labels) if label.startswith("delivery")]
        if delivery_indices:
            delivery_x = [all_x[i] for i in delivery_indices]
            delivery_y = [all_y[i] for i in delivery_indices]
            ax.scatter(delivery_x, delivery_y, c='red', s=120, marker='^', 
                      label='Delivery Stations', zorder=9, edgecolors='black', linewidths=1.5)
        
        # Plot optimized routes
        for route_idx, route in enumerate(self.routes):
            color = colors[route_idx % len(colors)]
            route_x = [all_x[node] for node in route]
            route_y = [all_y[node] for node in route]
            
            # Draw route path
            ax.plot(route_x, route_y, color=color, linewidth=2.5, 
                   alpha=0.7, zorder=5, label=f'Robot {route_idx + 1} Route')
            
            # Add arrows to show direction (show every 2-3 segments to avoid clutter)
            if show_arrows and len(route) > 1:
                arrow_interval = max(2, len(route) // 8)  # Show ~8 arrows per route
                for i in range(0, len(route) - 1, arrow_interval):
                    dx = route_x[i + 1] - route_x[i]
                    dy = route_y[i + 1] - route_y[i]
                    segment_length = np.sqrt(dx**2 + dy**2)
                    # Only show arrow if segment is long enough
                    if segment_length > 0.3:
                        # Position arrow at middle of segment
                        mid_x = route_x[i] + dx * 0.5
                        mid_y = route_y[i] + dy * 0.5
                        ax.annotate('', xy=(route_x[i + 1], route_y[i + 1]),
                                   xytext=(route_x[i], route_y[i]),
                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                  lw=2, alpha=0.9, shrinkA=5, shrinkB=5))
        
        # Set labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        problem_type = "TSP" if self.n_robots == 1 else f"VRP ({self.n_robots} robots)"
        ax.set_title(f'Optimized Routes - {problem_type}\n'
                    f'Total Distance: {self.total_distance:.2f} units', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Save optimization results to files.
        
        Args:
            output_dir: Directory to save the results
        
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save solution metadata
        metadata = {
            "n_robots": self.n_robots,
            "n_points": self.n_points,
            "depot_index": self.depot_index,
            "total_distance": float(self.total_distance),
            "solve_time": self.solve_time,
            "n_routes": len(self.routes),
            "route_distances": [float(d) for d in self.route_distances]
        }
        
        metadata_path = os.path.join(output_dir, "optimization_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save routes as CSV
        routes_data = []
        for route_idx, route in enumerate(self.routes):
            for node_idx, node in enumerate(route):
                routes_data.append({
                    "route_id": route_idx,
                    "node_order": node_idx,
                    "point_index": node,
                    "point_label": self.point_labels[node],
                    "x": self.points[node][0],
                    "y": self.points[node][1],
                    "route_distance": self.route_distances[route_idx] if route_idx < len(self.route_distances) else 0
                })
        
        routes_df = pd.DataFrame(routes_data)
        routes_path = os.path.join(output_dir, "optimized_routes.csv")
        routes_df.to_csv(routes_path, index=False)
        
        return {
            "metadata": metadata_path,
            "routes": routes_path
        }
    
    def print_summary(self) -> None:
        """Print a summary of the optimization results."""
        print("=" * 60)
        print("ROUTE OPTIMIZATION SUMMARY")
        print("=" * 60)
        problem_type = "TSP" if self.n_robots == 1 else f"VRP ({self.n_robots} robots)"
        print(f"Problem Type: {problem_type}")
        print(f"Number of Points: {self.n_points}")
        print(f"Depot Index: {self.depot_index}")
        
        if self.solution:
            print(f"\nTotal Route Length: {self.total_distance:.2f} units")
            print(f"Time to Solve: {self.solve_time:.3f} seconds")
            print(f"Number of Routes: {len(self.routes)}")
            
            if self.n_robots > 1:
                print("\nDistance per Robot:")
                for i, dist in enumerate(self.route_distances):
                    print(f"  Robot {i + 1}: {dist:.2f} units")
            
            print("\nRoute Details:")
            for i, route in enumerate(self.routes):
                route_labels = [self.point_labels[node] for node in route]
                print(f"  Robot {i + 1}: {' -> '.join(route_labels)}")
        else:
            print("\nNo solution found.")
        
        print("=" * 60)


def load_warehouse_data(data_dir: str = "data/raw") -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
    """
    Load warehouse data from data/raw/ directory.
    
    Args:
        data_dir: Directory containing the warehouse data files
    
    Returns:
        Tuple of (distance_matrix, points, point_labels)
    """
    # Load distance matrix
    matrix_path = os.path.join(data_dir, "distance_matrix.npy")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Distance matrix not found at {matrix_path}")
    distance_matrix = np.load(matrix_path)
    
    # Load points
    points_path = os.path.join(data_dir, "warehouse_points.csv")
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Points file not found at {points_path}")
    points_df = pd.read_csv(points_path)
    
    points = [(row['x'], row['y']) for _, row in points_df.iterrows()]
    point_labels = points_df['point_id'].tolist()
    
    return distance_matrix, points, point_labels


def main():
    """
    Main function to run route optimization.
    """
    # Load warehouse data
    print("Loading warehouse data...")
    try:
        distance_matrix, points, point_labels = load_warehouse_data()
        print(f"Loaded {len(points)} points")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run simulate_warehouse.py first to generate the data.")
        return
    
    # Choose problem type
    n_robots = 1  # Change to >1 for VRP
    problem_type = "TSP" if n_robots == 1 else "VRP"
    
    print(f"\nSolving {problem_type} problem with {n_robots} robot(s)...")
    
    # Create optimizer
    optimizer = RouteOptimizer(
        distance_matrix=distance_matrix,
        points=points,
        point_labels=point_labels,
        n_robots=n_robots,
        depot_index=0  # Start from first robot
    )
    
    # Solve
    result = optimizer.solve(
        search_strategy="PATH_CHEAPEST_ARC",
        time_limit_seconds=30
    )
    
    # Print summary
    optimizer.print_summary()
    
    # Save results
    print("\nSaving results to data/processed/...")
    saved_files = optimizer.save_results()
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")
    
    # Visualize
    print("\nGenerating visualization...")
    optimizer.visualize(
        save_path="data/processed/optimized_routes.png",
        show_arrows=True
    )
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()

