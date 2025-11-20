"""
Warehouse Simulation Module

This module generates a 2D grid warehouse simulation with:
- Random placement of robots, pick-up points, and delivery stations
- Distance matrix computation (Manhattan or Euclidean)
- Data export to data/raw/
- Simple visualization of the warehouse layout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import json
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import compute_distance_matrix, euclidean_distance, manhattan_distance


class WarehouseSimulator:
    """
    Simulates a 2D grid warehouse with robots, pick-up points, and delivery stations.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        n_robots: int = 3,
        n_pickup_points: int = 10,
        n_delivery_stations: int = 5,
        distance_type: str = "euclidean",
        random_seed: int = 42
    ):
        """
        Initialize the warehouse simulator.
        
        Args:
            grid_size: Tuple (width, height) of the warehouse grid
            n_robots: Number of robots in the warehouse
            n_pickup_points: Number of pick-up points (where packages are collected)
            n_delivery_stations: Number of delivery stations (where packages are delivered)
            distance_type: Type of distance metric ("euclidean" or "manhattan")
            random_seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.n_robots = n_robots
        self.n_pickup_points = n_pickup_points
        self.n_delivery_stations = n_delivery_stations
        self.distance_type = distance_type
        self.random_seed = random_seed
        
        # Initialize random state
        np.random.seed(random_seed)
        
        # Storage for generated points
        self.robots: List[Tuple[float, float]] = []
        self.pickup_points: List[Tuple[float, float]] = []
        self.delivery_stations: List[Tuple[float, float]] = []
        self.all_points: List[Tuple[float, float]] = []
        self.distance_matrix: np.ndarray = None
        
    def generate_positions(self) -> None:
        """
        Generate random positions for robots, pick-up points, and delivery stations.
        Ensures no two points are at the exact same location.
        """
        width, height = self.grid_size
        all_positions = set()
        
        # Generate robot positions
        self.robots = []
        for _ in range(self.n_robots):
            while True:
                pos = (np.random.uniform(0, width), np.random.uniform(0, height))
                if pos not in all_positions:
                    all_positions.add(pos)
                    self.robots.append(pos)
                    break
        
        # Generate pick-up point positions
        self.pickup_points = []
        for _ in range(self.n_pickup_points):
            while True:
                pos = (np.random.uniform(0, width), np.random.uniform(0, height))
                if pos not in all_positions:
                    all_positions.add(pos)
                    self.pickup_points.append(pos)
                    break
        
        # Generate delivery station positions
        self.delivery_stations = []
        for _ in range(self.n_delivery_stations):
            while True:
                pos = (np.random.uniform(0, width), np.random.uniform(0, height))
                if pos not in all_positions:
                    all_positions.add(pos)
                    self.delivery_stations.append(pos)
                    break
        
        # Combine all points in order: robots, pickups, deliveries
        self.all_points = self.robots + self.pickup_points + self.delivery_stations
    
    def compute_distances(self) -> None:
        """
        Compute the pairwise distance matrix for all points.
        """
        if len(self.all_points) == 0:
            raise ValueError("Positions must be generated before computing distances. Call generate_positions() first.")
        
        self.distance_matrix = compute_distance_matrix(self.all_points, self.distance_type)
    
    def get_point_labels(self) -> List[str]:
        """
        Get labels for all points (robot_0, pickup_0, delivery_0, etc.)
        
        Returns:
            List of point labels
        """
        labels = []
        labels.extend([f"robot_{i}" for i in range(self.n_robots)])
        labels.extend([f"pickup_{i}" for i in range(self.n_pickup_points)])
        labels.extend([f"delivery_{i}" for i in range(self.n_delivery_stations)])
        return labels
    
    def save_data(self, output_dir: str = "data/raw") -> Dict[str, str]:
        """
        Save warehouse data to files in data/raw/.
        
        Args:
            output_dir: Directory to save the data files
        
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "grid_size": self.grid_size,
            "n_robots": self.n_robots,
            "n_pickup_points": self.n_pickup_points,
            "n_delivery_stations": self.n_delivery_stations,
            "distance_type": self.distance_type,
            "random_seed": self.random_seed,
            "total_points": len(self.all_points)
        }
        
        metadata_path = os.path.join(output_dir, "warehouse_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save points as CSV
        labels = self.get_point_labels()
        points_df = pd.DataFrame({
            "point_id": labels,
            "x": [p[0] for p in self.all_points],
            "y": [p[1] for p in self.all_points],
            "type": (["robot"] * self.n_robots + 
                     ["pickup"] * self.n_pickup_points + 
                     ["delivery"] * self.n_delivery_stations)
        })
        
        points_path = os.path.join(output_dir, "warehouse_points.csv")
        points_df.to_csv(points_path, index=False)
        
        # Save distance matrix as CSV
        labels = self.get_point_labels()
        distance_df = pd.DataFrame(
            self.distance_matrix,
            index=labels,
            columns=labels
        )
        
        distance_path = os.path.join(output_dir, "distance_matrix.csv")
        distance_df.to_csv(distance_path)
        
        # Save distance matrix as numpy array
        matrix_path = os.path.join(output_dir, "distance_matrix.npy")
        np.save(matrix_path, self.distance_matrix)
        
        return {
            "metadata": metadata_path,
            "points": points_path,
            "distance_matrix_csv": distance_path,
            "distance_matrix_npy": matrix_path
        }
    
    def visualize(self, save_path: str = None, show_links: bool = True) -> None:
        """
        Create a simple visualization of the warehouse layout.
        
        Args:
            save_path: Optional path to save the figure
            show_links: If True, show lines connecting all points
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot robots
        if self.robots:
            robot_x = [p[0] for p in self.robots]
            robot_y = [p[1] for p in self.robots]
            ax.scatter(robot_x, robot_y, c='blue', s=150, marker='s', 
                      label=f'Robots ({self.n_robots})', zorder=5, edgecolors='black', linewidths=1.5)
        
        # Plot pick-up points
        if self.pickup_points:
            pickup_x = [p[0] for p in self.pickup_points]
            pickup_y = [p[1] for p in self.pickup_points]
            ax.scatter(pickup_x, pickup_y, c='green', s=100, marker='o', 
                      label=f'Pick-up Points ({self.n_pickup_points})', zorder=4, edgecolors='black', linewidths=1)
        
        # Plot delivery stations
        if self.delivery_stations:
            delivery_x = [p[0] for p in self.delivery_stations]
            delivery_y = [p[1] for p in self.delivery_stations]
            ax.scatter(delivery_x, delivery_y, c='red', s=100, marker='^', 
                      label=f'Delivery Stations ({self.n_delivery_stations})', zorder=4, edgecolors='black', linewidths=1)
        
        # Show links between all points (optional)
        if show_links and len(self.all_points) > 1:
            for i in range(len(self.all_points)):
                for j in range(i + 1, len(self.all_points)):
                    ax.plot([self.all_points[i][0], self.all_points[j][0]],
                           [self.all_points[i][1], self.all_points[j][1]],
                           'gray', alpha=0.22, linewidth=0.5, zorder=1)
        
        # Set labels and title
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'Warehouse Layout ({self.grid_size[0]}×{self.grid_size[1]})\n'
                    f'Distance Type: {self.distance_type.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Set axis limits with padding
        width, height = self.grid_size
        ax.set_xlim(-1, width + 1)
        ax.set_ylim(-1, height + 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate the generated warehouse data.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "matrix_not_empty": False,
            "matrix_coherent": False,
            "distances_positive": False,
            "diagonal_zero": False,
            "matrix_symmetric": False
        }
        
        if self.distance_matrix is None:
            return results
        
        # Check if matrix is not empty
        results["matrix_not_empty"] = self.distance_matrix.size > 0
        
        # Check if matrix is square
        n = len(self.all_points)
        results["matrix_coherent"] = self.distance_matrix.shape == (n, n)
        
        # Check if all distances are positive (except diagonal)
        mask = ~np.eye(n, dtype=bool)
        results["distances_positive"] = np.all(self.distance_matrix[mask] > 0)
        
        # Check if diagonal is zero
        results["diagonal_zero"] = np.all(np.diag(self.distance_matrix) == 0)
        
        # Check if matrix is symmetric (for euclidean/manhattan, should be symmetric)
        results["matrix_symmetric"] = np.allclose(self.distance_matrix, self.distance_matrix.T)
        
        return results
    
    def print_summary(self) -> None:
        """Print a summary of the warehouse simulation."""
        print("=" * 60)
        print("WAREHOUSE SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Grid Size: {self.grid_size[0]}×{self.grid_size[1]}")
        print(f"Robots: {self.n_robots}")
        print(f"Pick-up Points: {self.n_pickup_points}")
        print(f"Delivery Stations: {self.n_delivery_stations}")
        print(f"Total Points: {len(self.all_points)}")
        print(f"Distance Type: {self.distance_type}")
        print(f"Random Seed: {self.random_seed}")
        
        if self.distance_matrix is not None:
            print(f"\nDistance Matrix Shape: {self.distance_matrix.shape}")
            print(f"Min Distance: {np.min(self.distance_matrix[self.distance_matrix > 0]):.2f}")
            print(f"Max Distance: {np.max(self.distance_matrix):.2f}")
            print(f"Mean Distance: {np.mean(self.distance_matrix[self.distance_matrix > 0]):.2f}")
            
            # Validation results
            validation = self.validate()
            print("\nValidation Results:")
            for key, value in validation.items():
                status = "[OK]" if value else "[FAIL]"
                print(f"  {status} {key.replace('_', ' ').title()}")
        
        print("=" * 60)


def main():
    """
    Main function to run the warehouse simulation.
    """
    # Create simulator
    simulator = WarehouseSimulator(
        grid_size=(20, 20),
        n_robots=3,
        n_pickup_points=10,
        n_delivery_stations=5,
        distance_type="euclidean",
        random_seed=42
    )
    
    # Generate positions
    print("Generating warehouse positions...")
    simulator.generate_positions()
    
    # Compute distances
    print("Computing distance matrix...")
    simulator.compute_distances()
    
    # Print summary
    simulator.print_summary()
    
    # Save data
    print("\nSaving data to data/raw/...")
    saved_files = simulator.save_data()
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")
    
    # Visualize
    print("\nGenerating visualization...")
    simulator.visualize(save_path="data/raw/warehouse_layout.png", show_links=True)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

