"""
ML Data Generation Module

This module generates synthetic datasets for training ML models to predict
travel time, priority, or congestion in warehouse environments.

Features simulated:
- Congestion (robot density)
- Obstacles (blocked cells)
- Variable speed conditions
- Different warehouse configurations
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import os
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import euclidean_distance, manhattan_distance


class MLDataGenerator:
    """
    Generates synthetic ML datasets for warehouse route optimization.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        n_samples: int = 5000,
        random_seed: int = 42
    ):
        """
        Initialize the ML data generator.
        
        Args:
            grid_size: Tuple (width, height) of the warehouse grid
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_obstacles(self, obstacle_density: float = 0.1) -> List[Tuple[int, int]]:
        """
        Generate random obstacles (blocked cells) in the warehouse.
        
        Args:
            obstacle_density: Fraction of grid cells that are blocked (0.0 to 1.0)
        
        Returns:
            List of (x, y) coordinates of blocked cells
        """
        width, height = self.grid_size
        total_cells = width * height
        n_obstacles = int(total_cells * obstacle_density)
        
        obstacles = []
        obstacle_set = set()
        
        for _ in range(n_obstacles):
            while True:
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                if (x, y) not in obstacle_set:
                    obstacle_set.add((x, y))
                    obstacles.append((x, y))
                    break
        
        return obstacles
    
    def calculate_congestion_level(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        n_robots: int,
        grid_size: Tuple[int, int]
    ) -> float:
        """
        Calculate congestion level based on robot density in the area.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            n_robots: Number of robots in the warehouse
            grid_size: Size of the warehouse grid
        
        Returns:
            Congestion level (0.0 to 1.0)
        """
        # Calculate area of the route corridor
        distance = euclidean_distance(start, end)
        corridor_width = 2.0  # Assume 2-unit wide corridor
        area = distance * corridor_width
        
        # Total warehouse area
        width, height = grid_size
        total_area = width * height
        
        # Robot density in the corridor area
        # Higher density = more congestion
        base_density = n_robots / total_area
        corridor_density = (n_robots * area) / total_area
        
        # Normalize to 0-1 range
        congestion = min(1.0, corridor_density * 2.0)  # Scale factor for visibility
        
        # Add some randomness to simulate dynamic congestion
        congestion += np.random.uniform(-0.1, 0.1)
        congestion = max(0.0, min(1.0, congestion))
        
        return congestion
    
    def calculate_travel_time(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        distance: float,
        congestion: float,
        has_obstacles: bool,
        base_speed: float = 1.0
    ) -> float:
        """
        Calculate travel time considering distance, congestion, and obstacles.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            distance: Euclidean distance between points
            congestion: Congestion level (0.0 to 1.0)
            has_obstacles: Whether the path has obstacles
            base_speed: Base speed of the robot (units per time unit)
        
        Returns:
            Travel time (always >= distance / base_speed)
        """
        # Base travel time (distance / speed)
        base_time = distance / base_speed
        
        # Congestion penalty: slows down by up to 50%
        congestion_factor = 1.0 + (congestion * 0.5)
        
        # Obstacle penalty: adds 20% time if obstacles present
        obstacle_factor = 1.2 if has_obstacles else 1.0
        
        # Random variation (simulating real-world unpredictability)
        noise = np.random.uniform(0.95, 1.05)
        
        # Calculate final travel time
        travel_time = base_time * congestion_factor * obstacle_factor * noise
        
        # Ensure travel_time >= minimum (distance / max_speed)
        min_time = distance / (base_speed * 1.5)  # Maximum speed is 1.5x base
        travel_time = max(travel_time, min_time)
        
        return travel_time
    
    def check_path_has_obstacles(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstacles: List[Tuple[int, int]],
        threshold: float = 1.5
    ) -> bool:
        """
        Check if the path between start and end is near any obstacles.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            obstacles: List of obstacle coordinates
            threshold: Distance threshold to consider obstacle "on path"
        
        Returns:
            True if path is near obstacles, False otherwise
        """
        if not obstacles:
            return False
        
        # Check if any obstacle is close to the line segment
        for obstacle in obstacles:
            obs_x, obs_y = obstacle
            
            # Calculate distance from obstacle to line segment
            # Using point-to-line distance formula
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            if dx == 0 and dy == 0:
                # Start and end are the same
                dist = euclidean_distance(start, (obs_x, obs_y))
            else:
                # Project obstacle onto line segment
                t = max(0, min(1, ((obs_x - start[0]) * dx + (obs_y - start[1]) * dy) / (dx*dx + dy*dy)))
                proj_x = start[0] + t * dx
                proj_y = start[1] + t * dy
                dist = euclidean_distance((obs_x, obs_y), (proj_x, proj_y))
            
            if dist < threshold:
                return True
        
        return False
    
    def generate_dataset(
        self,
        n_robots_range: Tuple[int, int] = (1, 10),
        obstacle_density_range: Tuple[float, float] = (0.0, 0.2),
        balance_congestion: bool = True
    ) -> pd.DataFrame:
        """
        Generate a synthetic ML dataset.
        
        Args:
            n_robots_range: Range of number of robots (min, max)
            obstacle_density_range: Range of obstacle density (min, max)
            balance_congestion: If True, ensure balanced congestion levels
        
        Returns:
            DataFrame with columns: start_x, start_y, end_x, end_y, distance, 
                                   congestion, has_obstacles, predicted_time
        """
        width, height = self.grid_size
        data = []
        
        # Generate balanced congestion levels if requested
        if balance_congestion:
            # Create balanced distribution of congestion levels
            congestion_levels = np.linspace(0.0, 1.0, 10)  # 10 levels
            congestion_samples = np.random.choice(congestion_levels, size=self.n_samples)
        else:
            congestion_samples = np.random.uniform(0.0, 1.0, size=self.n_samples)
        
        for i in range(self.n_samples):
            # Generate random start and end points
            start_x = np.random.uniform(0, width)
            start_y = np.random.uniform(0, height)
            end_x = np.random.uniform(0, width)
            end_y = np.random.uniform(0, height)
            
            start = (start_x, start_y)
            end = (end_x, end_y)
            
            # Calculate distance
            distance = euclidean_distance(start, end)
            
            # Skip if start and end are too close (not meaningful)
            if distance < 0.5:
                continue
            
            # Generate random number of robots for this sample
            n_robots = np.random.randint(n_robots_range[0], n_robots_range[1] + 1)
            
            # Get congestion level (balanced or random)
            congestion = congestion_samples[i]
            
            # Generate obstacles for this configuration
            obstacle_density = np.random.uniform(obstacle_density_range[0], obstacle_density_range[1])
            obstacles = self.generate_obstacles(obstacle_density)
            
            # Check if path has obstacles
            has_obstacles = self.check_path_has_obstacles(start, end, obstacles)
            
            # Calculate travel time
            travel_time = self.calculate_travel_time(
                start, end, distance, congestion, has_obstacles
            )
            
            # Additional features
            n_obstacles_near = sum(1 for obs in obstacles 
                                  if euclidean_distance(start, obs) < 5.0 or 
                                     euclidean_distance(end, obs) < 5.0)
            
            data.append({
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y,
                'distance': distance,
                'congestion': congestion,
                'has_obstacles': int(has_obstacles),
                'n_obstacles_near': n_obstacles_near,
                'n_robots': n_robots,
                'obstacle_density': obstacle_density,
                'predicted_time': travel_time
            })
        
        df = pd.DataFrame(data)
        
        # Validate: ensure travel_time >= minimum
        min_time = df['distance'] / 1.5  # Max speed is 1.5x base
        df['predicted_time'] = df[['predicted_time', 'distance']].apply(
            lambda row: max(row['predicted_time'], row['distance'] / 1.5), axis=1
        )
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str = "data/raw") -> str:
        """
        Save the generated dataset to CSV.
        
        Args:
            df: DataFrame to save
            output_dir: Directory to save the file
        
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_dir, "ml_dataset.csv")
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def print_dataset_summary(self, df: pd.DataFrame) -> None:
        """Print a summary of the generated dataset."""
        print("=" * 60)
        print("ML DATASET SUMMARY")
        print("=" * 60)
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns) - 1}")  # Excluding target
        print(f"\nColumns: {', '.join(df.columns)}")
        
        print("\nDataset Statistics:")
        print(df.describe())
        
        print("\nCongestion Distribution:")
        print(df['congestion'].value_counts().sort_index().head(10))
        
        print("\nObstacles Distribution:")
        print(f"Paths with obstacles: {df['has_obstacles'].sum()} ({100*df['has_obstacles'].mean():.1f}%)")
        print(f"Paths without obstacles: {(~df['has_obstacles'].astype(bool)).sum()} ({100*(1-df['has_obstacles'].mean()):.1f}%)")
        
        print("\nValidation:")
        min_time = df['distance'] / 1.5
        invalid = (df['predicted_time'] < min_time).sum()
        print(f"Valid samples (travel_time >= min): {len(df) - invalid} ({100*(len(df)-invalid)/len(df):.1f}%)")
        if invalid > 0:
            print(f"WARNING: {invalid} samples have invalid travel_time")
        
        print("=" * 60)


def main():
    """
    Main function to generate ML dataset.
    """
    print("Generating ML dataset for warehouse route optimization...")
    
    # Create generator
    generator = MLDataGenerator(
        grid_size=(20, 20),
        n_samples=5000,  # Generate 5000 samples
        random_seed=42
    )
    
    # Generate dataset
    print("\nGenerating synthetic data...")
    df = generator.generate_dataset(
        n_robots_range=(1, 10),
        obstacle_density_range=(0.0, 0.2),
        balance_congestion=True
    )
    
    # Print summary
    generator.print_dataset_summary(df)
    
    # Save dataset
    print("\nSaving dataset to data/raw/...")
    output_path = generator.save_dataset(df)
    print(f"Dataset saved to: {output_path}")
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()

